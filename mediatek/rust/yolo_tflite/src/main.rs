use tflite::{FlatBufferModel, InterpreterBuilder};
use ndarray::{Array4, ArrayD};

struct TFLiteBackend {
    // Sử dụng lifetime 'static vì model được leak để đảm bảo tồn tại suốt thời gian chạy.
    interpreter: tflite::Interpreter<'static, tflite::ops::builtin::BuiltinOpResolver>,
    input_shape: Vec<usize>,
}

impl TFLiteBackend {
    /// Khởi tạo backend từ file model.
    pub fn new(model_path: &str) -> Self {
        // Load model và leak để có lifetime 'static.
        let model_box = Box::new(
            FlatBufferModel::build_from_file(model_path)
                .expect("Không thể load model")
        );
        let model_ref: &'static FlatBufferModel = Box::leak(model_box);

        // Sử dụng resolver mặc định.
        let resolver = tflite::ops::builtin::BuiltinOpResolver::default();
        let builder = InterpreterBuilder::new(model_ref, resolver)
            .expect("Không thể tạo InterpreterBuilder");
        let mut interpreter = builder.build().expect("Không thể xây dựng interpreter");
        interpreter.allocate_tensors().expect("Không thể cấp phát tensor");

        // Lấy thông tin input (giả sử chỉ có 1 input).
        let input_idx = interpreter.inputs()[0];
        let input_shape = interpreter.tensor_info(input_idx)
            .expect("Không tìm thấy tensor info cho input")
            .dims
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>();

        println!("Input shape expected by the model: {:?}", input_shape);

        Self { interpreter, input_shape }
    }

    /// Hàm forward chạy inference cho input.
    /// Ở đây input là tensor dạng ndarray với shape [batch, channels, height, width].
    pub fn forward(&mut self, mut im: Array4<f32>) -> Vec<ArrayD<f32>> {
        if self.input_shape.len() == 4 {
            if self.input_shape[1] == 640 && self.input_shape[3] == 3 {
                im = im.permuted_axes([0, 2, 3, 1]);
            } else if self.input_shape[1] == 3 && self.input_shape[2] == 640 {
                // Đã đúng định dạng.
            } else {
                panic!("Unexpected input shape từ model: {:?}", self.input_shape);
            }
        } else {
            panic!("Input shape không có 4 chiều: {:?}", self.input_shape);
        }

        let input_idx = self.interpreter.inputs()[0];
        let input_tensor = self.interpreter.tensor_data_mut(input_idx)
            .expect("Không lấy được input tensor data");
        let im_slice = im.as_slice().expect("Không chuyển đổi được ndarray thành slice");
        input_tensor.copy_from_slice(im_slice);

        self.interpreter.invoke().expect("Lỗi khi chạy inference");

        let mut outputs = Vec::new();
        for &output_idx in self.interpreter.outputs().iter() {
            let output_data = self.interpreter.tensor_data(output_idx)
                .expect("Không lấy được output tensor data")
                .to_vec();
            let out_shape = self.interpreter.tensor_info(output_idx)
                .expect("Không tìm thấy tensor info cho output")
                .dims
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>();
            let output_array = ndarray::ArrayD::from_shape_vec(out_shape, output_data)
                .expect("Không chuyển đổi output sang ndarray");
            outputs.push(output_array);
        }

        outputs
    }
}

fn main() {
    let model_path = "models/yolo11n-pose_saved_model/model.tflite";
    let mut backend = TFLiteBackend::new(model_path);

    // Giả sử input có shape [1, 3, 640, 640] (float32).
    let dummy_input = Array4::<f32>::zeros((1, 3, 640, 640));

    // Khi chạy inference, nếu model yêu cầu op không được đăng ký, sẽ báo lỗi:
    // "Didn't find op for builtin opcode 'RESIZE_NEAREST_NEIGHBOR' version '3'"
    let outputs = backend.forward(dummy_input);
    for (i, output) in outputs.iter().enumerate() {
        println!("Output {}: shape {:?}", i, output.shape());
    }
}
