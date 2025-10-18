//! Individual wrapper types for each GGML quantization format
//! 
//! These wrappers allow each GGML type to be registered as a separate
//! QuantizedDType variant, enabling users to specify exact quantization types.

use crate::quantized::ggml_quantized::{GGMLQuantized, GGMLQuantizedType};
use crate::Result;

macro_rules! define_ggml_wrapper {
    ($name:ident, $qtype:ident) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;
        
        impl $name {
            pub const NAME: &'static str = stringify!($name);
            
            #[inline]
            pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<()> {
                // Data includes header with type byte
                GGMLQuantized::dequantize(data, output)
            }
            
            #[inline]
            pub fn quantize(input: &[f32]) -> Result<Vec<u8>> {
                GGMLQuantized::quantize_with_type(input, GGMLQuantizedType::$qtype)
            }
            
            #[inline]
            pub fn storage_size_in_bytes(num_elements: usize) -> usize {
                GGMLQuantized::storage_size_in_bytes(num_elements)
            }
            
            #[inline]
            pub fn matmul(
                lhs_f32: &[f32],
                lhs_shape: &[usize],
                rhs_data: &[u8],
                rhs_shape: &[usize],
            ) -> Result<Vec<f32>> {
                GGMLQuantized::matmul(lhs_f32, lhs_shape, rhs_data, rhs_shape)
            }
        }
    };
}

// Define all 12 GGML quantization type wrappers
define_ggml_wrapper!(Q4_0, Q4_0);
define_ggml_wrapper!(Q4_1, Q4_1);
define_ggml_wrapper!(Q5_0, Q5_0);
define_ggml_wrapper!(Q5_1, Q5_1);
define_ggml_wrapper!(Q8_0, Q8_0);
define_ggml_wrapper!(Q8_1, Q8_1);
define_ggml_wrapper!(Q2K, Q2K);
define_ggml_wrapper!(Q3K, Q3K);
define_ggml_wrapper!(Q4K, Q4K);
define_ggml_wrapper!(Q5K, Q5K);
define_ggml_wrapper!(Q6K, Q6K);
define_ggml_wrapper!(Q8K, Q8K);
