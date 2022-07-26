from typing import Any, Optional, Callable, Tuple
import torch
import os
import gzip
import pickle
from Models import Vocabulary
from torchvision.datasets import VisionDataset


class SignGlossSample:

    def __init__(self, name, singer, glosses, words, signs_frames):
        self.name = name
        self.singer = singer
        self.glosses = torch.tensor(glosses, dtype=torch.int)
        self.words = torch.tensor(words, dtype=torch.int)
        self.signs_frames = signs_frames + 1e-8  # stability


class SignGlossLanguage(VisionDataset):
    source_url = "http://cihancamgoz.com/files/cvpr2020"
    train_files_list = [("train", "phoenix14t.pami0.train"),
                        ("dev", "phoenix14t.pami0.dev")]
    test_files_list = [("test", "phoenix14t.pami0.test"), ]

    def __init__(self,
                 root: str,
                 gloss_vocab: Vocabulary.GlossVocabulary,
                 word_vocab: Vocabulary.WordVocabulary,
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
        self.data = self._parser_data(gloss_vocab.get_stoi(), word_vocab.get_stoi())

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        sample = self.data[index]
        glosses = sample.glosses
        target = sample.words
        padding = torch.zeros(self.max_signs_frames - sample.signs_frames.shape[0], self.frame_size)
        video = torch.cat([sample.signs_frames, padding], axis=0)
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

    def _parser_data(self, gloss_to_idx, word_to_idx):
        samples = {}
        for _, source_file in self.source_files_list:
            with gzip.open(os.path.join(self.root, source_file), "rb") as f:
                loaded_object = pickle.load(f)
                for frame in loaded_object:
                    video_name = frame["name"]
                    if video_name in samples.keys():
                        raise RuntimeError(f"Please check {video_name}")
                    sample = SignGlossSample(
                        name=video_name,
                        singer=frame["signer"],
                        glosses=[gloss_to_idx[g] for g in frame["gloss"].strip().split(' ')],
                        words=[word_to_idx[w] for w in frame["text"].strip().split(' ')],
                        signs_frames=frame["sign"]
                        )
                    samples[video_name] = sample
                    self.max_glosses = max(self.max_glosses, len(sample.glosses))
                    self.max_words = max(self.max_words, len(sample.words))
                    self.max_signs_frames = max(self.max_signs_frames, sample.signs_frames.shape[0])

        # Padding
        for sample in samples.values():
            padding = torch.tensor([gloss_to_idx[Vocabulary.PAD_TOKEN]],
                                   dtype=torch.int).repeat(self.max_glosses - len(sample.glosses))
            sample.glosses = torch.cat((sample.glosses, padding), axis=0)
            padding = torch.tensor([word_to_idx[Vocabulary.PAD_TOKEN]],
                                   dtype=torch.int).repeat(self.max_words - len(sample.words))
            sample.words = torch.cat((sample.words, padding), axis=0)

        return list(samples.values())
