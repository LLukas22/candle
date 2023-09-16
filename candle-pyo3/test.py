import candle
from candle import Tensor, QTensor

t = candle.Tensor(42.0)
print(t)
print(t.shape, t.rank, t.device)
print(t + t)

t = candle.Tensor([3.0, 1, 4, 1, 5, 9, 2, 6])
print(t)
print(t+t)

t:Tensor = t.reshape([2, 4])
print(t.matmul(t.t()))

print(t.to_dtype(candle.u8))
print(t.to_dtype("u8"))

t = candle.randn((5, 3))
print(t)
print(t.dtype)

t = candle.randn((16, 256))
quant_t:QTensor = t.quantize("q6k")
dequant_t:Tensor = quant_t.dequantize()
diff2:Tensor = (t - dequant_t).sqr()
print(diff2.mean_all())
