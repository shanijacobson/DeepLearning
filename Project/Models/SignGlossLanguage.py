from typing import Any, Optional, Callable, Tuple
import torch
import os
import gzip
import pickle
import Project.Models.Vocabulary as vocab

from torch import Tensor
from torchvision.datasets import VisionDataset


class SignGlossSample:

    def __init__(self, name, singer, gloss, text):
        self.name = name
        self.singer = singer
        self.glosses = gloss.strip().split(' ')
        self.words = text.strip().split(' ')
        self.signs_frames: Tensor = torch.Tensor([])

    def add_sign_frame(self, frame):
        if frame["name"] != self.name:
            raise RuntimeError(f"Frame name is different then sample name.")
        if frame["signer"] != self.singer or frame["gloss"].strip().split(' ') != self.glosses or frame["text"].strip().split(' ') != self.words:
            raise RuntimeError(f"Sample {self.name} has some mishmash. Please check.")
        new_frame = frame["sign"] + 1e-8  # stability
        self.signs_frames = torch.cat([new_frame, self.signs_frames], axis=1)


class SignGlossLanguage(VisionDataset):
    source_url = "http://cihancamgoz.com/files/cvpr2020"
    train_files_list = [("train", "phoenix14t.pami0.train"),
                        ("dev", "phoenix14t.pami0.dev")]
    test_files_list = [("test", "phoenix14t.pami0.test"), ]

    def __init__(self,
                 root: str,
                 train: bool = True,
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 max_signs_frames=0,
                 max_glosses=0,
                 max_words=0) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.train = train
        self.frame_size = 1024
        self.max_signs_frames = max_signs_frames
        self.max_glosses = max_glosses
        self.max_words = max_words
        self.source_files_list = self.train_files_list if self.train else self.test_files_list

        if download:
            self._download_data()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        samples = {}
        for name, source_file in self.source_files_list:
            with gzip.open(os.path.join(self.root, source_file), "rb") as f:
                loaded_object = pickle.load(f)
                for frame in loaded_object:
                    if frame["name"] not in samples.keys():
                        samples[frame['name']] = SignGlossSample(
                            name=frame["name"],
                            singer=frame["signer"],
                            gloss=frame["gloss"],
                            text=frame["text"]
                        )
                    sample = samples[frame["name"]]
                    sample.add_sign_frame(frame)
                    self.max_glosses = max(self.max_glosses, len(sample.glosses))
                    self.max_words = max(self.max_words, len(sample.words))
                    self.max_signs_frames = max(self.max_signs_frames, sample.signs_frames.shape[0])
        self.data = list(samples.values())

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        sample = self.data[index]
        glosses = sample.glosses + [vocab.PAD_TOKEN] * (self.max_glosses - len(sample.glosses))
        target = sample.words + [vocab.PAD_TOKEN] * (self.max_words - len(sample.words))
        video = torch.cat([sample.signs_frames, torch.zeros(self.max_signs_frames - sample.signs_frames.shape[0], self.frame_size)], axis=0)
        if self.transform is not None:
            video = self.transform(video)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return video, glosses, target

    def __len__(self) -> int:
        return len(self.data)

    def _download_data(self):
        if self._check_integrity():
            print("Files already downloaded and verified")

        for file in self.source_files_list:
            if not os.path.exists(os.path.join(self.root, file[1])):
                os.system(f"wget '{self.source_url}/{file[1]}' -P {self.root}")

    def _check_integrity(self):
        return all(os.path.exists(os.path.join(self.root, file)) for _, file in self.source_files_list)



