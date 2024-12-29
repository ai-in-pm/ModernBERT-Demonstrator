import pytest
import torch
from app.models.modern_bert import ModernBERT, GeGLU, RotaryPositionalEmbedding

@pytest.fixture
def model():
    return ModernBERT(
        vocab_size=1000,
        max_seq_len=512,
        dim=256,
        depth=6,
        num_heads=8
    )

def test_model_initialization(model):
    assert isinstance(model, ModernBERT)
    assert model.max_seq_len == 512
    assert model.get_num_params() > 0

def test_geglu_activation():
    geglu = GeGLU(256, 512)
    x = torch.randn(2, 10, 256)
    output = geglu(x)
    assert output.shape == (2, 10, 512)

def test_rotary_embeddings():
    rope = RotaryPositionalEmbedding(64, max_seq_len=512)
    x = torch.randn(2, 100, 8, 64)
    cos, sin = rope(x, seq_len=100)
    assert cos.shape[1] == 100
    assert sin.shape[1] == 100

def test_model_forward_pass(model):
    batch_size = 2
    seq_length = 128
    x = torch.randint(0, 1000, (batch_size, seq_length))
    
    output = model(x)
    assert output.shape == (batch_size, seq_length, 1000)

def test_sequence_length_validation(model):
    with pytest.raises(ValueError):
        x = torch.randint(0, 1000, (2, 1000))  # seq_length > max_seq_len
        model(x)

def test_model_device_movement(model):
    if torch.cuda.is_available():
        model = model.cuda()
        assert next(model.parameters()).is_cuda
