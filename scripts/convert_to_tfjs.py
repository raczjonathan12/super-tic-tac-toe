"""One-off conversion of final_model.keras to a TF.js layers-model, written
by hand because the `tensorflowjs` pip package's dependency chain (via
tensorflow_decision_forests -> pandas) does not import successfully on this
machine's Python version. Produces the same model.json + weight shard
structure tf.loadLayersModel() expects."""
import json
import struct

from tensorflow import keras

model = keras.models.load_model("final_model.keras")

topology = json.loads(model.to_json())


def normalize_keras3_config(node):
    """tf.js's layers deserializer expects the pre-Keras-3 (TF 2.x/Keras 2)
    JSON shape: dtype as a plain string, InputLayer's shape key named
    batch_input_shape. Keras 3's to_json() instead emits a nested
    DTypePolicy object for dtype and renames the shape key to batch_shape.
    Recursively rewrite both back to the old shape so tf.js can load it."""
    if isinstance(node, dict):
        if (
            "dtype" in node
            and isinstance(node["dtype"], dict)
            and node["dtype"].get("class_name", "").endswith("DTypePolicy")
        ):
            node["dtype"] = node["dtype"]["config"]["name"]
        if "batch_shape" in node:
            node["batch_input_shape"] = node.pop("batch_shape")
        for value in node.values():
            normalize_keras3_config(value)
    elif isinstance(node, list):
        for item in node:
            normalize_keras3_config(item)


normalize_keras3_config(topology)
topology["keras_version"] = "2.15.0"
topology["backend"] = "tensorflow"


def convert_inbound_nodes(inbound_nodes):
    """tf.js's layers deserializer expects the pre-Keras-3 node-history
    format: a list of calls, each call a list of [layer_name, node_index,
    tensor_index, kwargs] tuples. Keras 3 instead serializes each call as
    {"args": [...__keras_tensor__ objects, possibly nested in a list for
    multi-input layers...], "kwargs": {...}}. Rebuild the old shape from the
    keras_history embedded in each tensor placeholder."""
    result = []
    for call in inbound_nodes:
        args = call.get("args", [])
        kwargs = call.get("kwargs") or {}
        input_specs = []

        def collect(node):
            if isinstance(node, list):
                for item in node:
                    collect(item)
            elif isinstance(node, dict) and node.get("class_name") == "__keras_tensor__":
                layer_name, node_index, tensor_index = node["config"]["keras_history"]
                input_specs.append([layer_name, node_index, tensor_index, {}])

        for a in args:
            collect(a)
        if input_specs and kwargs:
            input_specs[0][3] = kwargs
        result.append(input_specs)
    return result


for layer in topology["config"]["layers"]:
    if layer.get("inbound_nodes"):
        layer["inbound_nodes"] = convert_inbound_nodes(layer["inbound_nodes"])

weights_meta = []
weight_buffers = []
for layer in model.layers:
    for w in layer.weights:
        name = w.name.split(":")[0]
        if "/" not in name:
            name = f"{layer.name}/{name}"
        arr = w.numpy().astype("<f4")
        weights_meta.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float32",
        })
        weight_buffers.append(arr.tobytes())

manifest = {
    "format": "layers-model",
    "generatedBy": "scripts/convert_to_tfjs.py",
    "convertedBy": None,
    "modelTopology": topology,
    "weightsManifest": [{
        "paths": ["group1-shard1of1.bin"],
        "weights": weights_meta,
    }],
}

with open("docs/model/model.json", "w") as f:
    json.dump(manifest, f)

with open("docs/model/group1-shard1of1.bin", "wb") as f:
    for buf in weight_buffers:
        f.write(buf)

total_bytes = sum(len(b) for b in weight_buffers)
print(f"Wrote docs/model/model.json and group1-shard1of1.bin ({total_bytes} bytes, {len(weights_meta)} weights)")
