# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import AbstractDataset

__all__ = ["FONT_FAMILIES", "FONT_STYLE_ATTRIBUTES", "FontStyleDataset", "encode_font_style"]

# The font families known by the pretrained font style classifiers
FONT_FAMILIES: list[str] = [
    "Abhaya Libre",
    "Abyssinica SIL",
    "Afacad",
    "Agbalumo",
    "Aksaramatee",
    "Akshar",
    "Alata",
    "Alatsi",
    "Allura",
    "Almarai",
    "Amatic SC",
    "Amiko",
    "Amiri",
    "Anek Devanagari",
    "Anek Gujarati",
    "Anek Kannada",
    "Anek Malayalam",
    "Anek Odia",
    "Anek Tamil",
    "Annapurna SIL",
    "Annie Use Your Telescope",
    "Anuphan",
    "Aparajita",
    "Architects Daughter",
    "Aref Ruqaa",
    "Arial",
    "Arima",
    "Arimo",
    "Arya",
    "Asar",
    "Assistant",
    "Athiti",
    "Atma",
    "Azeret Mono",
    "BIZ UDPGothic",
    "BIZ UDPMincho",
    "BPG Arial",
    "Bad Script",
    "Bai Jamjuree",
    "Baloo 2",
    "Baloo Bhai 2",
    "Baloo Bhaina 2",
    "Baloo Chettan 2",
    "Baloo Paaji 2",
    "Baloo Thambi 2",
    "Battambang",
    "Beiruti",
    "Bellefair",
    "Bellota",
    "Bellota Text",
    "Biryani",
    "Black Han Sans",
    "BlockKie",
    "Bokor",
    "Bricolage Grotesque",
    "Cairo",
    "Caladea",
    "Calistoga",
    "Cambay",
    "Carlito",
    "Catamaran",
    "Caveat",
    "Caveat Brush",
    "Cedarville Cursive",
    "Chakra Petch",
    "Charm",
    "Charmonman",
    "Chilanka",
    "Chonburi",
    "Code2000",
    "Coiny",
    "Comic Neue",
    "Coming Soon",
    "Content",
    "Cookie",
    "Cormorant Garamond",
    "Cormorant Infant",
    "Cousine",
    "Covered By Your Grace",
    "Crafty Girls",
    "Dancing Script",
    "Dangrek",
    "David Libre",
    "Dawning of a New Day",
    "DejaVu Sans",
    "DejaVu Sans Mono",
    "DejaVu Serif",
    "Dekko",
    "Denk One",
    "Do Hyeon",
    "DotGothic16",
    "EB Garamond",
    "Eczar",
    "El Messiri",
    "Faculty Glyphic",
    "Fahkwang",
    "Fira Sans",
    "Fira Sans Condensed",
    "Fira Sans Extra Condensed",
    "Fjalla One",
    "Fragment Mono",
    "Frank Ruhl Libre",
    "Freehand",
    "Freeman",
    "Fustat",
    "Gaegu",
    "Galada",
    "Gamja Flower",
    "Gayathri",
    "Geist",
    "Geist Mono",
    "Gelasio",
    "Gemunu Libre",
    "Gidugu",
    "Glegoo",
    "Gloock",
    "Gloria Hallelujah",
    "Gochi Hand",
    "Gothic A1",
    "Gotu",
    "Gowun Batang",
    "Gowun Dodum",
    "Great Vibes",
    "Hahmlet",
    "Halant",
    "Handlee",
    "Hanuman",
    "Harmattan",
    "Heebo",
    "Herr Von Muellerhoff",
    "Hi Melody",
    "Hina Mincho",
    "Hind",
    "Hind Guntur",
    "Hind Madurai",
    "Hind Siliguri",
    "Hind Vadodara",
    "Homemade Apple",
    "IBM Plex Mono",
    "IBM Plex Sans",
    "IBM Plex Sans Arabic",
    "IBM Plex Sans Condensed",
    "IBM Plex Sans Devanagari",
    "IBM Plex Sans Hebrew",
    "IBM Plex Sans KR",
    "IBM Plex Sans Thai",
    "IBM Plex Sans Thai Looped",
    "IBM Plex Serif",
    "Inclusive Sans",
    "Indie Flower",
    "Inknut Antiqua",
    "Inter",
    "Inter Tight",
    "Itim",
    "Jaini",
    "Jaldi",
    "Javanese Text",
    "Jua",
    "Just Another Hand",
    "K2D",
    "Kadwa",
    "Kalam",
    "Kanit",
    "Kantumruy Pro",
    "Karma",
    "Katibeh",
    "Kelvinch",
    "Khand",
    "Khula",
    "Kiwi Maru",
    "Klee One",
    "KoHo",
    "Kodchasan",
    "Koh Santepheap",
    "Koulen",
    "Kristi",
    "Krub",
    "Kurale",
    "LXGW WenKai Mono TC",
    "LXGW WenKai TC",
    "La Belle Aurore",
    "Laila",
    "Lateef",
    "Latin Modern Mono",
    "Latin Modern Roman",
    "Latin Modern Sans",
    "Lato",
    "Liu Jian Mao Cao",
    "Lohit Marathi",
    "Long Cang",
    "Lora",
    "M PLUS 1p",
    "Ma Shan Zheng",
    "Mada",
    "Maitree",
    "Mali",
    "Manjari",
    "Marck Script",
    "Markazi Text",
    "Martel",
    "Martel Sans",
    "Meera Inimai",
    "Merriweather",
    "Mirza",
    "Mitr",
    "Mona Sans",
    "Montserrat",
    "Moul",
    "Mr Dafoe",
    "Mukta",
    "Mukta Malar",
    "Mukta Vaani",
    "Murecho",
    "Nanum Brush Script",
    "Nanum Pen",
    "NanumGothic",
    "NanumMyeongjo",
    "National Park",
    "Neucha",
    "Niramit",
    "Nokora",
    "Nothing You Could Do",
    "Noto Kufi Arabic",
    "Noto Naskh Arabic",
    "Noto Rashi Hebrew",
    "Noto Sans",
    "Noto Sans Arabic",
    "Noto Sans Armenian",
    "Noto Sans Bengali",
    "Noto Sans Devanagari",
    "Noto Sans Ethiopic",
    "Noto Sans Georgian",
    "Noto Sans Gujarati",
    "Noto Sans Gurmukhi",
    "Noto Sans Hebrew",
    "Noto Sans JP",
    "Noto Sans Javanese",
    "Noto Sans KR",
    "Noto Sans Kannada",
    "Noto Sans Khmer",
    "Noto Sans Lao",
    "Noto Sans Lao Looped",
    "Noto Sans Malayalam",
    "Noto Sans Mono",
    "Noto Sans Myanmar",
    "Noto Sans Oriya",
    "Noto Sans SC",
    "Noto Sans Sinhala",
    "Noto Sans Sundanese",
    "Noto Sans Tamil",
    "Noto Sans Telugu",
    "Noto Sans Thai",
    "Noto Sans Thai Looped",
    "Noto Serif",
    "Noto Serif Armenian",
    "Noto Serif Armenian Condensed",
    "Noto Serif Armenian ExtraCondensed",
    "Noto Serif Armenian SemiCondensed",
    "Noto Serif Bengali",
    "Noto Serif Devanagari",
    "Noto Serif Ethiopic",
    "Noto Serif Georgian",
    "Noto Serif Gujarati",
    "Noto Serif Gurmukhi",
    "Noto Serif Hebrew",
    "Noto Serif JP",
    "Noto Serif KR",
    "Noto Serif Kannada",
    "Noto Serif Khmer",
    "Noto Serif Lao",
    "Noto Serif Malayalam",
    "Noto Serif Myanmar",
    "Noto Serif Oriya",
    "Noto Serif SC",
    "Noto Serif Sinhala",
    "Noto Serif Tamil",
    "Noto Serif Telugu",
    "Noto Serif Thai",
    "Odor Mean Chey",
    "Ojuju",
    "Open Sans",
    "Over the Rainbow",
    "Overpass",
    "Overpass Mono",
    "Padauk",
    "Palanquin",
    "Palanquin Dark",
    "Pangolin",
    "Patrick Hand",
    "Pattaya",
    "Permanent Marker",
    "Phetsarath",
    "Platypi",
    "Playfair",
    "Playfair Display",
    "Playpen Sans",
    "Playpen Sans Arabic",
    "Playpen Sans Deva",
    "Playpen Sans Hebrew",
    "Playpen Sans Thai",
    "Playwrite GB S",
    "Plus Jakarta Sans",
    "Poppins",
    "Pragati Narrow",
    "Pridi",
    "Prompt",
    "REM",
    "Rajdhani",
    "Raleway",
    "Ramabhadra",
    "Rancho",
    "Ranga",
    "Recursive",
    "Reem Kufi",
    "Reenie Beanie",
    "Reggae One",
    "Rhodium Libre",
    "Roboto",
    "Roboto Mono",
    "Roboto Serif",
    "Rock Salt",
    "RocknRoll One",
    "Roman Unicode",
    "Rounded Mplus 1c",
    "Rozha One",
    "Rubik",
    "Sacramento",
    "Sahitya",
    "Sarabun",
    "Sarala",
    "Satisfy",
    "Sawarabi Mincho",
    "Scheherazade New",
    "Schibsted Grotesk",
    "Schoolbell",
    "Secular One",
    "Sedgwick Ave",
    "Shadows Into",
    "Shantell Sans",
    "Shippori Mincho",
    "Short Stack",
    "Shrikhand",
    "Song Myung",
    "Source Serif 4",
    "Space Grotesk",
    "Special Gothic",
    "Special Gothic Condensed One",
    "Sriracha",
    "Sue Ellen Francisco",
    "Suez One",
    "Sumana",
    "Sura",
    "Suwannaphum",
    "TH Sarabun New",
    "Tajawal",
    "Taprom",
    "Taviraj",
    "TeX Gyre Bonum",
    "TeX Gyre Cursor",
    "TeX Gyre Heros",
    "TeX Gyre Pagella",
    "TeX Gyre Schola",
    "TeX Gyre Termes",
    "Teko",
    "Thasadith",
    "The Girl Next Door",
    "Tillana",
    "Tinos",
    "Tiro Bangla",
    "Tiro Devanagari Hindi",
    "Tiro Gurmukhi",
    "Tiro Kannada",
    "Tiro Tamil",
    "Tiro Telugu",
    "Trirong",
    "Varela Round",
    "Vesper Libre",
    "Voltaire",
    "Waiting for the Sunrise",
    "Walter Turncoat",
    "Winky Rough",
    "Winky Sans",
    "Wittgenstein",
    "Xarrovv",
    "Yaldevi",
    "Yaldevi Colombo",
    "Yantramanav",
    "Yatra One",
    "Yellowtail",
    "Yomogi",
    "Yuji Syuku",
    "ZCOOL KuaiLe",
    "ZCOOL QingKe HuangYou",
    "Zen Kaku Gothic New",
    "Zen Kurenaido",
    "Zen Maru Gothic",
    "Zen Old Mincho",
    "Zeyada",
    "Zhi Mang Xing",
    "lipipragatuchhi",
    "站酷小薇体",
]

# The attributes predicted by the font style classifiers, in the order of the model outputs, each with its classes
FONT_STYLE_ATTRIBUTES: dict[str, list[Any]] = {
    "bold": [False, True],
    "italic": [False, True],
    "underline": ["none", "single", "double"],
    "strikethrough": [False, True],
    "color": ["black", "gray", "white", "red", "orange", "brown", "yellow", "green", "blue", "purple"],
    "highlight": ["none", "yellow", "green", "pink", "blue", "orange"],
    # Whether the word is written by hand (handwriting-style fonts included, typeset casual fonts excluded)
    "handwritten": [False, True],
    "category": ["sans_serif", "serif", "display", "handwriting", "monospace", "unknown"],
    "family": FONT_FAMILIES,
}


def _normalize(attribute: str, value: Any) -> Any:
    """Map a raw label value to its class name (e.g. a missing highlight to "none", "SANS_SERIF" to "sans_serif")"""
    if value is None:
        return "none"
    if attribute == "category" and isinstance(value, str):
        return value.lower()
    return value


def _value(label: dict[str, Any], attribute: str) -> Any:
    """The raw value of an attribute: `handwritten` defaults to whether the font category is handwriting"""
    if attribute == "handwritten" and attribute not in label and "category" in label:
        return _normalize("category", label["category"]) == "handwriting"
    if attribute not in label:
        raise KeyError(f"the label has no '{attribute}' attribute")
    return label[attribute]


def encode_font_style(label: dict[str, Any], attributes: dict[str, list[Any]] = FONT_STYLE_ATTRIBUTES) -> np.ndarray:
    """Encode the style label of a word crop as the class index of each attribute

    >>> from doctr.datasets import encode_font_style
    >>> encode_font_style({"bold": True, "italic": False, "underline": "double", "strikethrough": False,
    >>>                    "color": "red", "highlight": None, "category": "SERIF", "family": "Lora"})

    Args:
        label: the style attributes of the crop, e.g. {"bold": True, "underline": "single", "family": "Lato", ...}.
            Without a "handwritten" key, a word is handwritten when its font category is handwriting.
        attributes: the classes of each attribute, in the order of the model outputs

    Returns:
        an int64 array with the class index of each attribute
    """
    indices = []
    for attribute, classes in attributes.items():
        raw = _value(label, attribute)
        value = _normalize(attribute, raw)
        if value not in classes:
            raise ValueError(f"unknown value for the '{attribute}' attribute: {raw!r}")
        indices.append(classes.index(value))
    return np.asarray(indices, dtype=np.int64)


class FontStyleDataset(AbstractDataset):
    """Dataset implementation for font style classification of word crops: each target holds the class index
    of every style attribute (cf. `FONT_STYLE_ATTRIBUTES`)

    >>> from doctr.datasets import FontStyleDataset
    >>> train_set = FontStyleDataset(img_folder="/path/to/images",
    >>>                              labels_path="/path/to/labels.json")
    >>> sample = train_set[0]
    >>> img, target = sample.image, sample.target

    Args:
        img_folder: path to the images folder
        labels_path: path to the json file mapping each image name to its style attributes, e.g.
            {"img_1.jpg": {"bold": true, "italic": false, "underline": "none", "strikethrough": false,
            "color": "black", "highlight": null, "category": "SERIF", "family": "Lora"}, ...}.
            Other keys (e.g. "text") are ignored. Without a "handwritten" key, a word is handwritten when its font
            category is handwriting.
        attributes: the classes of each attribute, in the order of the model outputs
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        labels_path: str,
        attributes: dict[str, list[Any]] = FONT_STYLE_ATTRIBUTES,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, **kwargs)
        self.attributes = attributes

        with open(labels_path, encoding="utf-8") as f:
            labels = json.load(f)

        self.data: list[tuple[str, np.ndarray]] = []
        for img_name, label in labels.items():
            if not os.path.exists(os.path.join(self.root, img_name)):
                raise FileNotFoundError(f"unable to locate {os.path.join(self.root, img_name)}")
            try:
                target = encode_font_style(label, attributes)
            except (KeyError, ValueError) as e:
                raise ValueError(f"invalid label for {img_name}: {e}") from e
            self.data.append((img_name, target))

    def merge_dataset(self, ds: AbstractDataset) -> None:
        # Update data with new root for self (absolute paths, a relative one would be read from the new root)
        self.data = [(str(Path(self.root).absolute().joinpath(img_path)), label) for img_path, label in self.data]
        # Define new root
        self.root = Path("/")
        # Merge with ds data
        for img_path, label in ds.data:
            self.data.append((str(Path(ds.root).absolute().joinpath(img_path)), label))
