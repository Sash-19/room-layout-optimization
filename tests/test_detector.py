import numpy as np

from src.perception.detector import detections_from_result


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)

    def __len__(self):
        return len(self.xyxy.value)


class FakeMasks:
    def __init__(self, data):
        self.data = FakeTensor(data)


class FakeResult:
    def __init__(self, xyxy, conf, cls, names, masks=None, orig_shape=(4, 5)):
        self.boxes = FakeBoxes(xyxy, conf, cls)
        self.names = names
        self.masks = FakeMasks(masks) if masks is not None else None
        self.orig_shape = orig_shape


def test_detections_are_remapped_filtered_and_sorted():
    result = FakeResult(
        xyxy=[[1, 1, 5, 5], [2, 2, 6, 6], [3, 3, 7, 7]],
        conf=[0.4, 0.9, 0.8],
        cls=[2, 5, 9],
        names={2: "sofa", 5: "dining table", 9: "car"},
    )
    detections = detections_from_result(result)
    assert [item.cls_name for item in detections] == ["table", "couch"]
    assert detections[0].conf == 0.9
    assert detections[0].bbox == (2.0, 2.0, 6.0, 6.0)


def test_class_map_can_rename_and_drop():
    result = FakeResult(
        xyxy=[[0, 0, 4, 4], [1, 1, 3, 3]],
        conf=[0.7, 0.6],
        cls=[0, 1],
        names={0: "Office Chair", 1: "potted plant"},
    )
    detections = detections_from_result(result, class_map={"Office Chair": "chair", "potted plant": None})
    assert [item.cls_name for item in detections] == ["chair"]


def test_masks_are_thresholded_when_already_at_original_size():
    masks = np.array(
        [
            [[0.9, 0.1, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1, 0.1], [0.1] * 5, [0.1] * 5],
            [[0.1] * 5, [0.1] * 5, [0.6] * 5, [0.6] * 5],
        ],
        dtype=np.float32,
    )
    result = FakeResult(
        xyxy=[[0, 0, 5, 4], [0, 2, 5, 4]],
        conf=[0.8, 0.7],
        cls=[0, 1],
        names={0: "bed", 1: "chair"},
        masks=masks,
        orig_shape=(4, 5),
    )
    detections = detections_from_result(result)
    assert detections[0].mask.shape == (4, 5)
    assert detections[0].mask.sum() == 2
    assert detections[1].mask.sum() == 10


def test_empty_result():
    result = FakeResult(xyxy=np.zeros((0, 4)), conf=[], cls=[], names={})
    assert detections_from_result(result) == []
