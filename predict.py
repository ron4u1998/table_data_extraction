from mmdet.apis import init_detector, inference_detector, show_result_pyplot

checkpoint_file = '/home/samyak/table_data_extract/table_detection.pth'

config_file = 'default_runtime.py'

device='cpu'

model = init_detector(config_file, checkpoint_file, device=device)

result = inference_detector(model, 'input_images/input_2.jpg')

print(result)

show_result_pyplot(model, 'input_images/input_2.jpg' , result, out_file = 'model_output_images/output2.jpg')
