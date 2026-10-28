"""Export bge-m3 + bge-reranker-v2-m3 into the CPU runtime variants under test.

    python export_variants.py <src_models_dir> <out_dir> <calibration.json>

<src_models_dir> holds the pinned snapshots (bge-m3-5617a9f/,
bge-reranker-v2-m3-953dc6f/); <calibration.json> is an encoder_equivalence.py
bundle whose "texts" calibrate the full-int8 variant. Variants:

  ov-fp32        OpenVINO IR, fp32
  ov-int8w       OpenVINO, int8 WEIGHTS only (NNCF weight compression; fp32 compute)
  ov-int8        OpenVINO, int8 weights AND activations (NNCF PTQ, calibrated)
  ort-int8       ONNX Runtime, dynamic int8 quantization (AVX2 config)

The torch-fp32 baseline needs no export: server.py loads the snapshot directly.
"""
import json
import sys
from pathlib import Path

src, out, calib = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
MODELS = {"embed": "bge-m3-5617a9f", "rerank": "bge-reranker-v2-m3-953dc6f"}
texts = json.load(open(calib))["texts"]


def ov_class(role):
    from optimum.intel import OVModelForFeatureExtraction, OVModelForSequenceClassification
    return OVModelForFeatureExtraction if role == "embed" else OVModelForSequenceClassification


def calib_dataset(role, tok):
    from datasets import Dataset
    if role == "embed":
        rows = [{"text": t} for t in texts]
        return Dataset.from_list(rows).map(
            lambda b: tok(b["text"], truncation=True, max_length=512, padding="max_length"),
            batched=True, remove_columns=["text"])
    q = [t.split(".")[0][:180] for t in texts[:40]]
    pairs = [{"q": qq, "d": d} for qq in q for d in texts[:8]]
    return Dataset.from_list(pairs).map(
        lambda b: tok(b["q"], b["d"], truncation=True, max_length=512, padding="max_length"),
        batched=True, remove_columns=["q", "d"])


for role, name in MODELS.items():
    from transformers import AutoTokenizer
    path = src / name
    tok = AutoTokenizer.from_pretrained(path)
    cls = ov_class(role)

    d = out / "ov-fp32" / role
    if not d.exists():
        m = cls.from_pretrained(path, export=True); m.save_pretrained(d); tok.save_pretrained(d)
        print("wrote", d, flush=True)

    d = out / "ov-int8w" / role
    if not d.exists():
        from optimum.intel import OVWeightQuantizationConfig
        m = cls.from_pretrained(path, export=True, quantization_config=OVWeightQuantizationConfig(bits=8))
        m.save_pretrained(d); tok.save_pretrained(d)
        print("wrote", d, flush=True)

    d = out / "ov-int8" / role
    if not d.exists():
        from optimum.intel import OVQuantizer
        m = cls.from_pretrained(path, export=True)
        q = OVQuantizer.from_pretrained(m)
        q.quantize(calibration_dataset=calib_dataset(role, tok), save_directory=d)
        tok.save_pretrained(d)
        print("wrote", d, flush=True)

    d = out / "ort-int8" / role
    if not d.exists():
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        ocls = ORTModelForFeatureExtraction if role == "embed" else ORTModelForSequenceClassification
        tmp = out / "ort-fp32" / role
        if not tmp.exists():
            ocls.from_pretrained(path, export=True).save_pretrained(tmp); tok.save_pretrained(tmp)
        q = ORTQuantizer.from_pretrained(tmp)
        q.quantize(save_dir=d, quantization_config=AutoQuantizationConfig.avx2(is_static=False, per_channel=True))
        tok.save_pretrained(d)
        print("wrote", d, flush=True)
