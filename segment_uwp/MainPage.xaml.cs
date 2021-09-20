using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using Microsoft.AI.MachineLearning;
using Microsoft.AI.MachineLearning.Experimental;
using System.Diagnostics;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;
using Windows.Media;
using Windows.UI.Xaml.Media.Imaging;



// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace segment_uwp
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private LearningModel _learningModel = null;
        private LearningModelDeviceKind _inferenceDeviceSelected = LearningModelDeviceKind.Default;
        private LearningModelSession _session;
        private LearningModelBinding _binding;
        private string _modelSource = "fcn-resnet50-11.onnx";
        private bool _useGpu = false;
        private string _imgSource = "charlie.PNG";// "testimg.jpg";
        private string _inputImageDescription;
        private string _outputImageDescription;
        uint _inWidth, _inHeight, _outWidth, _outHeight;
        private BitmapDecoder _decoder = null;

        public MainPage()
        {
            this.InitializeComponent();
            LoadModelAsync();
            getImageAsync();
        }

        private void LoadModelAsync()
        {
            Debug.Write("LoadModelBegin | ");

            Debug.Write("LoadModel Lock | ");

            _binding?.Clear();
            _session?.Dispose();

            StorageFile modelFile = StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/{_modelSource}")).GetAwaiter().GetResult();
            _learningModel = LearningModel.LoadFromStorageFileAsync(modelFile).GetAwaiter().GetResult();
            _inferenceDeviceSelected = _useGpu ? LearningModelDeviceKind.DirectX : LearningModelDeviceKind.Cpu;

            _session = new LearningModelSession(_learningModel, new LearningModelDevice(_inferenceDeviceSelected));
            _binding = new LearningModelBinding(_session);

            debugModelIO();
            _inputImageDescription = _learningModel.InputFeatures.ToList().First().Name;
            _outputImageDescription = _learningModel.OutputFeatures.ToList().First().Name;

            Debug.Write("LoadModel Unlock\n");
        }

        public void debugModelIO()
        {
            string _inName, _outName;
            foreach (var inputF in _learningModel.InputFeatures)
            {
                TensorFeatureDescriptor tfDesc = inputF as TensorFeatureDescriptor;
                Debug.WriteLine($"input | kind:{inputF.Kind}, name:{inputF.Name}" +
                    $" Shape: {string.Join(",", tfDesc.Shape.ToArray<long>())}");
            }
            foreach (var outputF in _learningModel.OutputFeatures)
            {
                TensorFeatureDescriptor tfDesc = outputF as TensorFeatureDescriptor;
                Debug.WriteLine($"output | kind:{outputF.Kind}, name:{outputF.Name}" +
                    $" Shape: {string.Join(",", tfDesc.Shape.ToArray<long>())}");
            }
        }

        private async void UIButtonAcquireImage_Click(object sender, RoutedEventArgs e)
        {
            var file = StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/{_imgSource}")).GetAwaiter().GetResult();
            var stream = file.OpenAsync(FileAccessMode.Read).GetAwaiter().GetResult();
            var decoder = BitmapDecoder.CreateAsync(stream).GetAwaiter().GetResult();
            _inWidth = decoder.PixelWidth;
            _inHeight = decoder.PixelHeight;
            Debug.WriteLine("Loaded image");
        }

        public static LearningModel ArgMax(long axis, long h, long w)
        {
            var builder = LearningModelBuilder.Create(12)
                .Inputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("Data", TensorKind.Float, new long[] { -1, -1, h, w })) // Different input type? 
                .Outputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("Output", TensorKind.Float, new long[] { -1, -1, h, w })) // Output of int64? 
                .Operators.Add(new LearningModelOperator("ArgMax")
                    .SetInput("data", "Data")
                    .SetAttribute("keepdims", TensorInt64Bit.CreateFromArray(new List<long>(), new long[] { 1 }))
                    .SetAttribute("axis", TensorInt64Bit.CreateFromIterable(new long[] { }, new long[] { axis })) // Correct way of passing axis? 
                    .SetOutput("reduced", "Reduced"))
                .Operators.Add(new LearningModelOperator("Cast")
                    .SetInput("input", "Reduced")
                    .SetAttribute("to", TensorInt64Bit.CreateFromIterable(new long[] { }, new long[] { (long) TensorizationModels.OnnxDataType.FLOAT }))
                    .SetOutput("output", "Output"))
                ;

            return builder.CreateModel();
        }

        public static LearningModel GetForeground(long n, long c, long h, long w)
        {
            var builder = LearningModelBuilder.Create(12)
                .Inputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("InputImage", TensorKind.Float, new long[] { n, c, h, w }))
                .Inputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("InputMask", TensorKind.Float, new long[] { n, 1, h, w })) // Broadcast to each color channel
                .Inputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("InputBackground", TensorKind.Float, new long[] { n, c, h, w }))
                .Outputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("Output", TensorKind.Float, new long[] { n, c, h, w }))
                .Operators.Add(new LearningModelOperator("Clip")
                    .SetInput("input", "InputMask")
                    .SetConstant("max", TensorFloat.CreateFromIterable(new long[] { 1 }, new float[] { 1 }))
                    .SetOutput("output", "MaskBinary"))
                .Operators.Add(new LearningModelOperator("Mul")
                    .SetInput("A", "InputImage")
                    .SetInput("B", "MaskBinary")
                    .SetOutput("C", "Foreground"))
                .Operators.Add(new LearningModelOperator("Add")
                    .SetInput("A", "InputBackground")
                    .SetInput("B", "Foreground")
                    .SetOutput("C", "Output"))
                ;
                
            return builder.CreateModel();
        }

        public static LearningModel GetBackground(long n, long c, long h, long w)
        {
            var builder = LearningModelBuilder.Create(12)
                .Inputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("InputImage", TensorKind.Float, new long[] { n, c, h, w }))
                .Inputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("InputMask", TensorKind.Float, new long[] { n, 1, h, w })) // Broadcast to each color channel
                .Outputs.Add(LearningModelBuilder.CreateTensorFeatureDescriptor("Output", TensorKind.Float, new long[] { n, c, h, w }))
                .Operators.Add(new LearningModelOperator("Clip") // Make mask binary 
                    .SetInput("input", "InputMask")
                    .SetConstant("max", TensorFloat.CreateFromIterable(new long[] { 1 }, new float[] { 1 }))
                    .SetOutput("output", "ClipMask"))
                .Operators.Add(new LearningModelOperator("Mul") 
                    .SetInput("A", "ClipMask")
                    .SetConstant("B", TensorFloat.CreateFromIterable(new long[] { 1 }, new float[] { -1 }))
                    .SetOutput("C", "NegMask"))
                .Operators.Add(new LearningModelOperator("Add") // BackgroundMask = (1- InputMask)
                    .SetConstant("A", TensorFloat.CreateFromIterable(new long[] { 1 }, new float[] { 1 }))
                    .SetInput("B", "NegMask")
                    .SetOutput("C", "BackgroundMask"))
                .Operators.Add(new LearningModelOperator("Mul") // Extract the background
                    .SetInput("A", "InputImage")
                    .SetInput("B", "BackgroundMask")
                    .SetOutput("C", "Output"))
                ;


            return builder.CreateModel();
        }

        public async void getImageAsync()
        {
            Debug.WriteLine("In GetImage");
            StorageFile file = StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/testimg.jpg")).GetAwaiter().GetResult();
            using (IRandomAccessStream stream = file.OpenAsync(FileAccessMode.Read).GetAwaiter().GetResult())
            {
                Debug.WriteLine("Got stream");
                //BitmapDecoder decoder = BitmapDecoder.CreateAsync(stream).GetAwaiter().GetResult(); //needs to step through

                BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);
                Debug.WriteLine("Got decoder");

                _inWidth = decoder.PixelWidth;
                _inHeight = decoder.PixelHeight;

                var pixelDataProvider = await decoder.GetPixelDataAsync();//.GetAwaiter().GetResult();
                Debug.WriteLine("Got pixeldata");

                var bytes = pixelDataProvider.DetachPixelData();
                var buffer = bytes.AsBuffer(); // Does this do a copy??
                var inputRawTensor = TensorUInt8Bit.CreateFromBuffer(new long[] { 1, buffer.Length }, buffer);

                // 3 channel NCHW
                var nextOutputShape = new long[] { 1, 3, _inHeight, _inWidth };
                var tensorizedImg = TensorFloat.Create(nextOutputShape); // Need to keep this intermediate for blur/bckground

                // *** Reshape initial input
                LearningModelSession tensorizationSession = new LearningModelSession(TensorizationModels.ReshapeFlatBufferToNCHW(1, 4, _inHeight, _inWidth));
                var tensorizationBinding = Evaluate(tensorizationSession, inputRawTensor, tensorizedImg);
                Debug.WriteLine($"Intermediate Shape: {string.Join(",", tensorizedImg.Shape.ToArray<long>())}");

                // *** Normalize
                float[] mean = new float[] { 0.485f, 0.456f, 0.406f };
                float[] std = new float[] { 0.229f, 0.224f, 0.225f };
                // Already sliced out alpha, but normalize0_1 expects the input to still have it- could just write another version later
                LearningModelSession normalizeSession = new LearningModelSession(TensorizationModels.Normalize0_1ThenZScore(_inHeight, _inWidth, 4, mean, std));
                var intermediateTensor = TensorFloat.Create(tensorizedImg.Shape);
                var normalizationBinding = Evaluate(normalizeSession, tensorizedImg, intermediateTensor);

                // *** Run through actual model
                var modelOutputShape = new long[] { 1, 21, _inHeight, _inWidth };
                var modelOutputTensor = TensorFloat.Create(modelOutputShape);
                var modelBinding = Evaluate(_session, intermediateTensor, modelOutputTensor);

                // *** Get the class predictions for each pixel
                var rawLabels = TensorFloat.Create(new long[] { 1, 1, _inHeight, _inWidth });
                LearningModelSession labelsSession = new LearningModelSession(ArgMax(1, _inHeight, _inWidth));
                var labelsBinding = Evaluate(labelsSession, modelOutputTensor, rawLabels);
                //rawLabels.GetAsVectorView().Where(x => x > 0 );
                //Debug.WriteLine(String.Join(", ",rawLabels.GetAsVectorView().Where(x => x> 0 ).ToArray()));

                // Clip the mask to {0,1}
                /*intermediateTensor = TensorFloat.Create(nextOutputShape);
                var clipSession = new LearningModelSession(ClipMask(1, 1, _inHeight, _inWidth));
                var clipBinding = Evaluate(clipSession, rawLabels, intermediateTensor);
                Debug.WriteLine(String.Join(", ", intermediateTensor.GetAsVectorView().Where(x => x > 0 ).ToArray())); */

                // Create a blurred version of the original picture
                intermediateTensor = TensorFloat.Create(nextOutputShape);
                var blurSession = new LearningModelSession(TensorizationModels.AveragePool(50));
                var blurBinding = Evaluate(blurSession, tensorizedImg, intermediateTensor);

                // *** Get just the background based on mask
                var blurredImg = TensorFloat.Create(nextOutputShape);
                var backgroundSession = new LearningModelSession(GetBackground(1, 3, _inHeight, _inWidth));
                var binding = new LearningModelBinding(backgroundSession);
                binding.Bind(backgroundSession.Model.InputFeatures[0].Name, intermediateTensor);
                binding.Bind(backgroundSession.Model.InputFeatures[1].Name, rawLabels);
                binding.Bind(backgroundSession.Model.OutputFeatures[0].Name, blurredImg);
                EvaluateInternal(backgroundSession, binding);

                // *** Get just the foreground based on mask
                intermediateTensor = TensorFloat.Create(nextOutputShape);
                var foregroundSession = new LearningModelSession(GetForeground(1, 3, _inHeight, _inWidth));
                binding = new LearningModelBinding(foregroundSession);
                binding.Bind(foregroundSession.Model.InputFeatures[0].Name, tensorizedImg);
                binding.Bind(foregroundSession.Model.InputFeatures[1].Name, rawLabels);
                binding.Bind(foregroundSession.Model.InputFeatures[2].Name, blurredImg);
                binding.Bind(foregroundSession.Model.OutputFeatures[0].Name, intermediateTensor);
                EvaluateInternal(foregroundSession, binding);
                //var foregroundBinding = Evaluate(foregroundSession, rawLabels, intermediateTensor);

                //** Detensorize and output
                var outputFrame = Detensorize(intermediateTensor);

                SoftwareBitmap displayBitmap = outputFrame.SoftwareBitmap;
                //Image control only accepts BGRA8 encoding and Premultiplied/no alpha channel. This checks and converts
                //the SoftwareBitmap we want to bind.
                if (displayBitmap.BitmapPixelFormat != BitmapPixelFormat.Bgra8 ||
                    displayBitmap.BitmapAlphaMode != BitmapAlphaMode.Premultiplied)
                {
                    displayBitmap = SoftwareBitmap.Convert(displayBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }

                

                // get software bitmap souce
                var source = new SoftwareBitmapSource();
                source.SetBitmapAsync(displayBitmap).GetAwaiter();
                // draw the input image
                InputImage.Source = source;
            }

        }       

        private VideoFrame Detensorize(TensorFloat intermediateTensor)
        {
            // Change from Int64 to float
            //ITensor interm = intermediateTensor; 
            var shape = intermediateTensor.Shape;
            var n = (int)shape[0];
            var c = (int)shape[1];
            var h = (int)shape[2];
            var w = (int)shape[3];

            // Rather than writing the data into the software bitmap ourselves from a Tensor (which may be on the gpu)
            // we call an indentity model to move the gpu memory back to the cpu via WinML de-tensorization.
            var outputImage = new SoftwareBitmap(BitmapPixelFormat.Bgra8, w, h, BitmapAlphaMode.Ignore);
            var outputFrame = VideoFrame.CreateWithSoftwareBitmap(outputImage);

            LearningModelSession detensorizationSession = new LearningModelSession(TensorizationModels.IdentityNCHW(1, c, _inHeight, _inWidth));
            var descriptor = detensorizationSession.Model.InputFeatures[0] as TensorFeatureDescriptor;
            var detensorizerShape = descriptor.Shape;
            /*if (c != detensorizerShape[1] || h != detensorizerShape[2] || w != detensorizerShape[3])
            {
                detensorizationSession = CreateLearningModelSession(TensorizationModels.IdentityNCHW(n, c, h, w));
            }*/
            var detensorizationBinding = Evaluate(detensorizationSession, intermediateTensor, outputFrame, true);
            return outputFrame;

        }

        private LearningModelBinding Evaluate(LearningModelSession session, object input, object output, bool wait = false)
        {
            // Create the binding
            var binding = new LearningModelBinding(session);

            // Bind inputs and outputs
            string inputName = session.Model.InputFeatures[0].Name;
            string outputName = session.Model.OutputFeatures[0].Name;
            binding.Bind(inputName, input);

            var outputBindProperties = new PropertySet();
            //outputBindProperties.Add("DisableTensorCpuSync", PropertyValue.CreateBoolean(true));
            binding.Bind(outputName, output, outputBindProperties);

            // Evaluate
            EvaluateInternal(session, binding, wait);

            return binding;
        }

        private void EvaluateInternal(LearningModelSession session, LearningModelBinding binding, bool wait = false)
        {
            if (!_useGpu) //originally isCpu
            {
                session.Evaluate(binding, "");
            }
            else
            {
                var results = session.EvaluateAsync(binding, "");
                if (wait)
                {
                    results.GetAwaiter().GetResult();
                }
            }
        }
    }

   
}
