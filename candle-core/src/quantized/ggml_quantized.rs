//! GGMLQuantized - A unified wrapper for all GGML quantization types
//!
//! This module provides a single type that can be used with the register_quantized_types! macro
//! to register all GGML quantization formats at once.

use super::k_quants::*;
use crate::Result;

/// GGML Quantization Type - wraps all GGML block types
/// This allows registering all GGML quantization formats with a single type.
/// 
/// # Usage
/// 
/// ```ignore
/// use candle_quantized_macros::register_quantized_types;
/// use candle_core::quantized::ggml_quantized::GGMLQuantized;
/// 
/// // Register all GGML types with a single macro call!
/// register_quantized_types!(GGMLQuantized);
/// ```
pub struct GGMLQuantized;

impl GGMLQuantized {
    /// The name to use for this quantized type
    pub const NAME: &'static str = "ggml";
    
    /// Dequantize GGML quantized data to f32 (CPU)
    /// 
    /// The first byte of `data` indicates which GGML quantization type to use.
    pub fn dequantize(data: &[u8], output: &mut [f32]) -> Result<()> {
        if data.is_empty() {
            crate::bail!("Empty data for GGML dequantization");
        }
        
        // First byte indicates the quantization type
        let qtype = data[0];
        
        // Skip header (padded to alignment)
        use crate::quantized::k_quants::BlockQ4K;
        let block_align = std::mem::align_of::<BlockQ4K>().max(8);
        let header_size = block_align;
        
        if data.len() < header_size {
            crate::bail!("Data too small for GGML header");
        }
        
        let quant_data = &data[header_size..];
        
        let ggml_type = GGMLQuantized::decode_type(qtype)?;
        ggml_type.dequantize(quant_data, output)
    }
    
    /// Quantize f32 data to GGML quantized format (CPU)
    /// 
    /// By default, uses Q4_K quantization (good balance of quality/size).
    /// The output includes a 1-byte header indicating the quantization type.
    pub fn quantize(input: &[f32]) -> Result<Vec<u8>> {
        Self::quantize_with_type(input, GGMLQuantizedType::Q4K)
    }
    
    /// Quantize with a specific GGML quantization type
    pub fn quantize_with_type(input: &[f32], qtype: GGMLQuantizedType) -> Result<Vec<u8>> {
        let quantized = qtype.quantize(input)?;
        
        // Create output with proper alignment
        // We need to ensure the quantized data after the header is properly aligned
        use crate::quantized::k_quants::BlockQ4K;
        let block_align = std::mem::align_of::<BlockQ4K>().max(8);
        let header_size = block_align; // Use alignment as header size to ensure next data is aligned
        
        let mut output = vec![0u8; header_size + quantized.len()];
        output[0] = qtype as u8; // Store type in first byte
        // Rest of header is padding (zeros)
        output[header_size..].copy_from_slice(&quantized);
        
        Ok(output)
    }
    
    /// Calculate storage size in bytes for a given number of elements
    /// 
    /// Uses Q4_K as the default quantization type for size estimation.
    pub fn storage_size_in_bytes(num_elements: usize) -> usize {
        // Header (padded to alignment) + quantized data size
        use crate::quantized::k_quants::BlockQ4K;
        let block_align = std::mem::align_of::<BlockQ4K>().max(8);
        let header_size = block_align;
        
        header_size + GGMLQuantizedType::Q4K.storage_size_in_bytes(num_elements)
    }
    
    /// Matrix multiplication: f32 × GGML quantized → f32
    /// 
    /// The first byte of `rhs_data` indicates which GGML quantization type to use.
    pub fn matmul(
        lhs_f32: &[f32],
        lhs_shape: &[usize],
        rhs_data: &[u8],
        rhs_shape: &[usize],
    ) -> Result<Vec<f32>> {
        if rhs_data.is_empty() {
            crate::bail!("Empty rhs_data for GGML matmul");
        }
        
        // First byte indicates the quantization type
        let qtype = rhs_data[0];
        
        // Skip header (padded to alignment)
        use crate::quantized::k_quants::BlockQ4K;
        let block_align = std::mem::align_of::<BlockQ4K>().max(8);
        let header_size = block_align;
        
        if rhs_data.len() < header_size {
            crate::bail!("RHS data too small for GGML header");
        }
        
        let quant_data = &rhs_data[header_size..];
        
        let ggml_type = GGMLQuantized::decode_type(qtype)?;
        ggml_type.matmul(lhs_f32, lhs_shape, quant_data, rhs_shape)
    }
    
    /// Decode a quantization type byte
    fn decode_type(byte: u8) -> Result<GGMLQuantizedType> {
        GGMLQuantizedType::from_u8(byte).ok_or_else(|| {
            crate::Error::Msg(format!("Invalid GGML quantization type: {}", byte))
        })
    }
}

/// GGML Quantization Type Enum
/// 
/// This enum represents all the different GGML quantization formats.
/// It's stored as a u8 in the quantized data header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GGMLQuantizedType {
    Q4_0 = 0,
    Q4_1 = 1,
    Q5_0 = 2,
    Q5_1 = 3,
    Q8_0 = 4,
    Q8_1 = 5,
    Q2K = 6,
    Q3K = 7,
    Q4K = 8,
    Q5K = 9,
    Q6K = 10,
    Q8K = 11,
}

impl GGMLQuantizedType {
    /// Convert from u8 byte
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::Q4_0),
            1 => Some(Self::Q4_1),
            2 => Some(Self::Q5_0),
            3 => Some(Self::Q5_1),
            4 => Some(Self::Q8_0),
            5 => Some(Self::Q8_1),
            6 => Some(Self::Q2K),
            7 => Some(Self::Q3K),
            8 => Some(Self::Q4K),
            9 => Some(Self::Q5K),
            10 => Some(Self::Q6K),
            11 => Some(Self::Q8K),
            _ => None,
        }
    }
    
    /// Get the name of this quantization type
    pub fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2K",
            Self::Q3K => "Q3K",
            Self::Q4K => "Q4K",
            Self::Q5K => "Q5K",
            Self::Q6K => "Q6K",
            Self::Q8K => "Q8K",
        }
    }
    
    /// Calculate storage size for a given number of elements
    pub fn storage_size_in_bytes(self, num_elements: usize) -> usize {
        let block_size = self.block_size();
        let num_blocks = num_elements.div_ceil(block_size);
        num_blocks * self.block_size_in_bytes()
    }
    
    /// Get the block size (number of elements per block)
    pub fn block_size(self) -> usize {
        match self {
            Self::Q4_0 => BlockQ4_0::BLCK_SIZE,
            Self::Q4_1 => BlockQ4_1::BLCK_SIZE,
            Self::Q5_0 => BlockQ5_0::BLCK_SIZE,
            Self::Q5_1 => BlockQ5_1::BLCK_SIZE,
            Self::Q8_0 => BlockQ8_0::BLCK_SIZE,
            Self::Q8_1 => BlockQ8_1::BLCK_SIZE,
            Self::Q2K => BlockQ2K::BLCK_SIZE,
            Self::Q3K => BlockQ3K::BLCK_SIZE,
            Self::Q4K => BlockQ4K::BLCK_SIZE,
            Self::Q5K => BlockQ5K::BLCK_SIZE,
            Self::Q6K => BlockQ6K::BLCK_SIZE,
            Self::Q8K => BlockQ8K::BLCK_SIZE,
        }
    }
    
    /// Get the size of one block in bytes
    pub fn block_size_in_bytes(self) -> usize {
        match self {
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
        }
    }
    
    /// Dequantize data to f32
    pub fn dequantize(self, data: &[u8], output: &mut [f32]) -> Result<()> {
        let num_blocks = output.len().div_ceil(self.block_size());
        let expected_size = num_blocks * self.block_size_in_bytes();
        
        if data.len() != expected_size {
            crate::bail!(
                "Invalid data size for {:?}: expected {} bytes, got {}",
                self,
                expected_size,
                data.len()
            );
        }
        
        // Dispatch to the appropriate block type
        macro_rules! dequantize_dispatch {
            ($block_type:ty) => {{
                let blocks = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const $block_type,
                        num_blocks,
                    )
                };
                <$block_type as GgmlType>::to_float(blocks, output);
            }};
        }
        
        match self {
            Self::Q4_0 => dequantize_dispatch!(BlockQ4_0),
            Self::Q4_1 => dequantize_dispatch!(BlockQ4_1),
            Self::Q5_0 => dequantize_dispatch!(BlockQ5_0),
            Self::Q5_1 => dequantize_dispatch!(BlockQ5_1),
            Self::Q8_0 => dequantize_dispatch!(BlockQ8_0),
            Self::Q8_1 => dequantize_dispatch!(BlockQ8_1),
            Self::Q2K => dequantize_dispatch!(BlockQ2K),
            Self::Q3K => dequantize_dispatch!(BlockQ3K),
            Self::Q4K => dequantize_dispatch!(BlockQ4K),
            Self::Q5K => dequantize_dispatch!(BlockQ5K),
            Self::Q6K => dequantize_dispatch!(BlockQ6K),
            Self::Q8K => dequantize_dispatch!(BlockQ8K),
        }
        
        Ok(())
    }
    
    /// Quantize f32 data to this format
    pub fn quantize(self, input: &[f32]) -> Result<Vec<u8>> {
        let num_blocks = input.len().div_ceil(self.block_size());
        let output_size = num_blocks * self.block_size_in_bytes();
        let mut output = vec![0u8; output_size];
        
        // Dispatch to the appropriate block type
        macro_rules! quantize_dispatch {
            ($block_type:ty) => {{
                let blocks = unsafe {
                    std::slice::from_raw_parts_mut(
                        output.as_mut_ptr() as *mut $block_type,
                        num_blocks,
                    )
                };
                <$block_type as GgmlType>::from_float(input, blocks);
            }};
        }
        
        match self {
            Self::Q4_0 => quantize_dispatch!(BlockQ4_0),
            Self::Q4_1 => quantize_dispatch!(BlockQ4_1),
            Self::Q5_0 => quantize_dispatch!(BlockQ5_0),
            Self::Q5_1 => quantize_dispatch!(BlockQ5_1),
            Self::Q8_0 => quantize_dispatch!(BlockQ8_0),
            Self::Q8_1 => quantize_dispatch!(BlockQ8_1),
            Self::Q2K => quantize_dispatch!(BlockQ2K),
            Self::Q3K => quantize_dispatch!(BlockQ3K),
            Self::Q4K => quantize_dispatch!(BlockQ4K),
            Self::Q5K => quantize_dispatch!(BlockQ5K),
            Self::Q6K => quantize_dispatch!(BlockQ6K),
            Self::Q8K => quantize_dispatch!(BlockQ8K),
        }
        
        Ok(output)
    }
    
    /// Matrix multiplication: f32 × quantized → f32
    /// This implements the GGML-style mixed precision matmul
    pub fn matmul(
        self,
        lhs_f32: &[f32],
        lhs_shape: &[usize],
        rhs_data: &[u8],
        rhs_shape: &[usize],
    ) -> Result<Vec<f32>> {
        // Extract dimensions (assuming standard matmul shape [m, k] × [k, n] = [m, n])
        if lhs_shape.len() < 2 || rhs_shape.len() < 2 {
            crate::bail!("matmul requires at least 2D shapes");
        }
        
        let m = lhs_shape[lhs_shape.len() - 2];
        let k = lhs_shape[lhs_shape.len() - 1];
        let n = rhs_shape[rhs_shape.len() - 1];
        
        // Verify shapes are compatible
        if rhs_shape[rhs_shape.len() - 2] != k {
            crate::bail!(
                "matmul shape mismatch: lhs [..., {}, {}] × rhs [..., {}, {}]",
                m,
                k,
                rhs_shape[rhs_shape.len() - 2],
                n
            );
        }
        
        let mut output = vec![0.0f32; m * n];
        
        // Dispatch to the appropriate matmul implementation
        macro_rules! matmul_dispatch {
            ($block_type:ty) => {{
                let k_in_blocks = k.div_ceil(<$block_type>::BLCK_SIZE);
                let expected_size = n * k_in_blocks * self.block_size_in_bytes();
                
                if rhs_data.len() != expected_size {
                    crate::bail!(
                        "Invalid rhs data size for {:?}: expected {} bytes, got {}",
                        self,
                        expected_size,
                        rhs_data.len()
                    );
                }
                
                let rhs_blocks = unsafe {
                    std::slice::from_raw_parts(
                        rhs_data.as_ptr() as *const $block_type,
                        n * k_in_blocks,
                    )
                };
                
                matmul::<$block_type>((m, k, n), lhs_f32, rhs_blocks, &mut output)?;
            }};
        }
        
        match self {
            Self::Q4_0 => matmul_dispatch!(BlockQ4_0),
            Self::Q4_1 => matmul_dispatch!(BlockQ4_1),
            Self::Q5_0 => matmul_dispatch!(BlockQ5_0),
            Self::Q5_1 => matmul_dispatch!(BlockQ5_1),
            Self::Q8_0 => matmul_dispatch!(BlockQ8_0),
            Self::Q8_1 => matmul_dispatch!(BlockQ8_1),
            Self::Q2K => matmul_dispatch!(BlockQ2K),
            Self::Q3K => matmul_dispatch!(BlockQ3K),
            Self::Q4K => matmul_dispatch!(BlockQ4K),
            Self::Q5K => matmul_dispatch!(BlockQ5K),
            Self::Q6K => matmul_dispatch!(BlockQ6K),
            Self::Q8K => matmul_dispatch!(BlockQ8K),
        }
        
        Ok(output)
    }
}
