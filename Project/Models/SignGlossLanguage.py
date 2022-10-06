from typing import Any, Optional, Callable, Tuple
import torch
import numpy as np
from torch import nn
import os
import gzip
import pickle
from Models import Vocabulary
from torchvision.datasets import VisionDataset
from PIL import Image
import torchvision.transforms as transforms

VIDEO_CURL_COMMAND = "curl 'https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz' -H 'Connection: keep-alive' --compressed -o data.zip"


class SignGlossLanguage(VisionDataset):
    source_url = "http://cihancamgoz.com/files/cvpr2020"
    files = {
        "train": "phoenix14t.pami0.train",
        "dev": "phoenix14t.pami0.dev",
        "test": "phoenix14t.pami0.test"
    }
    original_video_path = os.path.join("Video", "PHOENIX-2014-T-release-v3", "PHOENIX-2014-T",
                                       "features", "fullFrame-210x260px")

    def __init__(self,
                 root: str,
                 gloss_vocab: Vocabulary,
                 word_vocab: Vocabulary,
                 type: str = "train",
                 download: bool = False,
                 original_frames: bool = False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 max_signs_frames=0,
                 max_glosses=0,
                 max_words=0,
                 feature_path=None,
                 do_conv = True,
                 poses_flag = False) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.type = type
        self.original_frames = original_frames
        self.frame_size = 260 * 210 if self.original_frames else 1024
        self.num_of_channels = 3 if self.original_frames else 1
        self.max_signs_frames = max_signs_frames
        self.max_glosses = max_glosses
        self.max_words = max_words
        self.max_allowed_frames = 400  # maximum allowed len of frames

        if download:
            self._download_data()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        self.data = self._parser_data(gloss_vocab, word_vocab)

        if feature_path is not None:
            with open(feature_path, 'rb') as f:
                self.features = pickle.load(f)
                if poses_flag:
                    self._format_poses()

                self.features = {key: value for key, value in self.features.items() if len(value) > 0}
                if do_conv:
                    self._convolod_input()

                self.data = list(filter(lambda x: x["name"].split("/")[1] in self.features.keys(), self.data))

        else:
            self.features = None

    def _format_poses(self, dim=3):
        for key, value in self.features.items():
            results = []
            for frame in value:
                frame = [vec[:dim] for vec in frame]
                frame = np.array(frame)
                results.append(frame.reshape(frame.shape[0] * frame.shape[1]).astype(np.float32))
            self.features[key] = torch.tensor(results)

    def _convolod_input(self, kernel=[0.1, 0.2, 0.4, 0.8, 0.4, 0.2, 0.1]):
        conv = nn.Conv1d(1, 1, 3, stride=1, padding=0, bias=False)
        kernel = torch.tensor(kernel)
        kernel /= kernel.sum()
        conv.weight = torch.nn.Parameter(kernel.view(1, 1, -1))

        for path, values in list(self.features.items()):
            values = torch.tensor(values)
            if values.shape[0] >= len(kernel):
                values_for_conv = values.view(torch.tensor(values).shape[0], 1, -1).permute(2, 1, 0).float()
                temp = conv(values_for_conv).permute(2, 1, 0).squeeze().detach()
                conv_results = torch.concat([values[:len(kernel) - 1], temp])
                self.features[path] = conv_results

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        sample = self.data[index]
        glosses = sample["glosses"]
        words = sample["words"]
        frames = self._get_videos_frame(sample)
        if self.target_transform is not None:
            words = self.target_transform(words)
        if self.features is not None:
            features = torch.tensor(self.features[sample["name"].split("/")[1]])
            features = torch.concat([features, torch.ones([frames.shape[0] - features.shape[0], features.shape[1]])])
            return (frames, sample["frames_len"]), (glosses, sample["glosses_len"]), (
            words, sample["words_len"]), features

        return (frames, sample["frames_len"]), (glosses, sample["glosses_len"]), (words, sample["words_len"])

    def __len__(self) -> int:
        return len(self.data)

    def _get_videos_frame(self, sample):
        video = torch.Tensor()
        if self.original_frames:
            transform = transforms.ToTensor()
            path = os.path.join(self.root, self.original_video_path, sample["name"])
            for frame_file in os.listdir(path):
                video_frame = transform(Image.open(os.path.join(path, frame_file))).view(1, self.num_of_channels, -1)
                video = torch.cat((video, video_frame), dim=0)
        else:
            video = sample["signs_frames"].view(-1, self.num_of_channels, self.frame_size)
        padding = torch.zeros(self.max_signs_frames - video.shape[0], self.num_of_channels,
                              self.frame_size, dtype=torch.int)
        video = torch.cat((video, padding), dim=0)

        if self.transform is not None:
            video = self.transform(video)
        return video

    def _download_data(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
        if self.type not in self.files.keys():
            raise f"Type {self.type} does not exist"
        if not os.path.exists(os.path.join(self.root, self.files[self.type])):
            os.system(f"wget '{self.source_url}/{self.files[self.type]}' -P {self.root}")

        if self.original_frames:
            if not os.path.exists(os.path.join(self.root, "Video", "data.zip")):
                if VIDEO_CURL_COMMAND is None:
                    raise RuntimeError("Please add curl command to download the data.")
                os.system(VIDEO_CURL_COMMAND)

    def _check_integrity(self):
        article_data_exist = all(os.path.exists(os.path.join(self.root, file)) for file in self.files.values())
        video_data_exist = not self.original_frames or os.path.exists(
            os.path.join(self.root, "Video", "PHOENIX-2014-T-release-v3"))
        return article_data_exist & video_data_exist

    def _parser_data(self, gloss_vocab, word_vocab):
        path = os.path.join("Data", "models", f"{self.type}_dataset")
        if os.path.exists(path):
            print(f"Getting existing dataset: {path}")
            samples = torch.load(path)
            self.max_glosses = samples[0]['glosses'].size()
            self.max_words = samples[0]['words'].size()
            self.max_signs_frames = max([sample['signs_frames'].size(0) for sample in samples])
            return samples

        samples = {}

        with gzip.open(os.path.join(self.root, self.files[self.type]), "rb") as f:
            loaded_object = pickle.load(f)
            for example in loaded_object:
                video_name = example["name"]
                if video_name in samples.keys():
                    raise RuntimeError(f"Please check {video_name}")
                sign_frames = example["sign"] + 1e-8  # numerical stability
                if sign_frames.shape[0] > self.max_allowed_frames:
                    continue
                glosses = [gloss_vocab[g] for g in example["gloss"].strip().split(' ')]
                # glosses = [gloss_vocab[g] for gloss in glosses for g in gloss.split('+')]
                words = [word_vocab[Vocabulary.BOS_TOKEN]] + \
                        [word_vocab[w] for w in example["text"].strip().split(' ')] + \
                        [word_vocab[Vocabulary.EOS_TOKEN]]
                samples[video_name] = {"name": video_name,
                                       "singer": example["signer"],
                                       "glosses": torch.tensor(glosses, dtype=torch.int),
                                       "words": torch.tensor(words, dtype=torch.long),
                                       "signs_frames": torch.tensor(1) if self.original_frames else sign_frames,
                                       "glosses_len": 0,
                                       "words_len": 0,
                                       "frames_len": sign_frames.shape[0]}
                self.max_glosses = max(self.max_glosses, len(glosses))
                self.max_words = max(self.max_words, len(words))
                self.max_signs_frames = max(self.max_signs_frames, sign_frames.shape[0])

        # Padding
        for sample in samples.values():
            glosses_len = len(sample["glosses"])
            padding = torch.tensor([gloss_vocab[Vocabulary.PAD_TOKEN]],
                                   dtype=torch.int).repeat(self.max_glosses - glosses_len)
            sample["glosses"] = torch.cat((sample["glosses"], padding), dim=0)
            sample["glosses_len"] = glosses_len
            words_len = len(sample["words"])
            padding = torch.tensor([word_vocab[Vocabulary.PAD_TOKEN]],
                                   dtype=torch.long).repeat(self.max_words - words_len)
            sample["words"] = torch.cat((sample["words"], padding), dim=0)
            sample["words_len"] = words_len
        samples = list(samples.values())
        return samples
