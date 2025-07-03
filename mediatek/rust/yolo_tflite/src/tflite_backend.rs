use tflite::FlatBufferModel;
use tflite::InterpreterBuilder;
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::interpreter::types::Type;
use anyhow::{Result, bail};

pub struct TfliteBackend<'a> {
    interpreter: tflite::Interpreter<'a, &'a BuiltinOpResolver>,
    model: FlatBufferModel,
    model_path: String,
}

impl<'a> TfliteBackend<'a> {
    pub fn new(model_path: &str) -> Result<Self> {
        let model = FlatBufferModel::build_from_file(model_path)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let resolver = BuiltinOpResolver::default();
        let builder = InterpreterBuilder::new(&model, &resolver)?;
        let mut interpreter = builder.build()?;
        interpreter.allocate_tensors()?;

        Ok(Self {
            interpreter,
            model,
            model_path: model_path.to_string(),
        })
    }

    pub fn forward(&mut self, im: Vec<f32>, h: u32, w: u32) -> Result<Vec<Vec<f32>>> {
        let input_tensor_index = self.interpreter.inputs()[0];
        self.interpreter.copy(&im, input_tensor_index)?;

        self.interpreter.invoke()?;

        let mut outputs = Vec::new();
        for output_tensor_index in self.interpreter.outputs() {
            let output_data: Vec<f32> = self.interpreter.copy(output_tensor_index)?;
            outputs.push(output_data);
        }

        Ok(outputs)
    }
}