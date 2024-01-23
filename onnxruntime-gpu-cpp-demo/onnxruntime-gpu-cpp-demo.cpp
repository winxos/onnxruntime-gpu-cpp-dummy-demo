/*
onnxruntime gpu cpp dummy demo
winxos 20240123
*/
#include <iostream>  
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace std;
using namespace Ort;
using namespace cv::dnn;

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

int main()
{
	OrtEnv* env;
	g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
	OrtSessionOptions* session_options;
	g_ort->CreateSessionOptions(&session_options);
	g_ort->SetIntraOpNumThreads(session_options, 1);
	g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_DISABLE_ALL);
	
	OrtCUDAProviderOptions cuda_option; //CUDA option set
	cuda_option.device_id = 0;
	cuda_option.arena_extend_strategy = 0;
	cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	cuda_option.gpu_mem_limit = SIZE_MAX;
	cuda_option.do_copy_in_default_stream = 1;
	g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_option);

	const wchar_t* model_path = L"best.onnx";
	OrtSession* session;
	g_ort->CreateSession(env, model_path, session_options, &session);
	OrtAllocator* allocator;
	g_ort->GetAllocatorWithDefaultOptions(&allocator);

	std::vector<const char*> input_node_names = {"images"};
	std::vector<std::vector<int64_t>> input_node_dims = { { 1,3,224,224 } }; 
	std::vector<ONNXTensorElementDataType> input_types = { ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT }; 
	std::vector<OrtValue*> input_tensors(1); 
	std::vector<const char*> output_node_names = { "output0" };
	std::vector<std::vector<int64_t>> output_node_dims = { {1,3} };
	std::vector<OrtValue*> output_tensors(1);

	Mat img = imread("1.bmp");

	vector<int64_t> input_shape = { 1, 3, 224, 224 };
	cv::Scalar mean = cv::Scalar(0.5, 0.5, 0.5);
	cv::Scalar std = cv::Scalar(0.5, 0.5, 0.5);
	Mat image;
	cv::resize(img, image, Size(224, 224));
	cv::cvtColor(image, image, COLOR_BGR2RGB);
	image.convertTo(image, CV_32F);
	image /= 255.0;
	cv::subtract(image, mean, image);
	cv::divide(image, std, image);
	Mat blob;
	dnn::blobFromImage(image, blob);
	size_t input_data_length = blob.total() * blob.elemSize();
	puts("load success!");

	OrtMemoryInfo* memory_info;
	g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
	g_ort->CreateTensorWithDataAsOrtValue(
		memory_info, reinterpret_cast<void*>(blob.data), input_data_length,
		input_shape.data(), input_shape.size(), input_types[0], &input_tensors[0]);
	g_ort->ReleaseMemoryInfo(memory_info);

	clock_t startTime, endTime;
	//warmup
	g_ort->Run(session, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(), 
		input_tensors.size(), output_node_names.data(), output_node_names.size(), output_tensors.data());
	int n = 100;
	startTime = clock();
	for (int i = 0; i < n; i++)
	{
		g_ort->Run(session, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
			input_tensors.size(), output_node_names.data(), output_node_names.size(),output_tensors.data());
	}
	endTime = clock();
	cout <<"infer "<< n << "x used:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	void* output_buffer;
	g_ort->GetTensorMutableData(output_tensors[0], &output_buffer);
	float* float_buffer = reinterpret_cast<float*>(output_buffer);
	size_t output_data_size = output_node_dims[0][1];
	auto max = std::max_element(float_buffer, float_buffer + output_data_size);

	std::vector<float> optu(float_buffer, float_buffer + output_data_size);
	int max_index = static_cast<int>(std::distance(float_buffer, max));

	puts("done!");
    return 0;
}
