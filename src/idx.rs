//! Decode IDX_file_format
//! return `&` all the time

use std::{fmt::Display, fs::File, io::Read, usize};

use crate::common::Float;

///
/// all with same size
///
pub struct ImageSet {
    raw: Vec<Float>,
    len: usize,
    width: usize,
    height: usize,
}

impl ImageSet {
    ///
    /// For example, with three dimensions of size n1, n2 and n3,
    /// respectively, the resulting Matrix object will have n1 rows and n2×n3 columns.
    ///
    pub fn loads(path: &str) -> ImageSet {
        let idx = IdxData::load(path);
        assert!(idx.dimension() == 3, "support d3 image only");

        let len = idx.sizes[0];
        let width = idx.sizes[1];
        let height = idx.sizes[2];

        ImageSet {
            raw: idx.data.into_iter().map(|n| n as Float / 255.0).collect(),
            len,
            width,
            height,
        }
    }

    pub fn iter(&self) -> ImageSetIter {
        self.into_iter()
    }
}

pub struct ImageSetIter<'a> {
    inner: &'a ImageSet,
    i: usize,
}

impl<'a> Iterator for ImageSetIter<'a> {
    type Item = ImageData<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.inner.len {
            return None;
        }

        let img_len = self.inner.width * self.inner.height;
        let start = self.i * img_len;
        let end = (self.i + 1) * img_len;
        let img_data = &self.inner.raw[start..end];

        let img = ImageData {
            data: img_data,
            width: self.inner.width,
            height: self.inner.height,
        };
        self.i += 1;

        Some(img)
    }
}

impl<'a> IntoIterator for &'a ImageSet {
    type Item = ImageData<'a>;
    type IntoIter = ImageSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        ImageSetIter { inner: &self, i: 0 }
    }
}

///
/// 0 ~ 9
///
pub struct LabelSet {
    data: Vec<u8>,
}

impl LabelSet {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn loads(path: &str) -> LabelSet {
        let idx = IdxData::load(path);
        assert_eq!(idx.dimension(), 1);

        LabelSet { data: idx.data }
    }

    pub fn iter(&self) -> LabelSetIter {
        self.into_iter()
    }
}

impl<'a> IntoIterator for &'a LabelSet {
    type Item = u8;

    type IntoIter = LabelSetIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        LabelSetIter { inner: &self, i: 0 }
    }
}

pub struct LabelSetIter<'a> {
    inner: &'a LabelSet,
    i: usize,
}

impl<'a> Iterator for LabelSetIter<'a> {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.inner.len() {
            return None;
        }

        let n = self.inner.data[self.i];
        self.i += 1;
        Some(n)
    }
}

///
/// grayscale image
///
pub struct ImageData<'a> {
    data: &'a [Float],
    width: usize,
    height: usize,
}

impl<'a> ImageData<'a> {
    fn color_at(&'a self, x: usize, y: usize) -> Float {
        self.data[y * self.height + x]
    }

    pub fn all_pixels(&'a self) -> &'a [Float] {
        self.data
    }
}

impl<'a> Display for ImageData<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                let ch = if self.color_at(x, y) > 0.0 { '#' } else { '_' };
                write!(f, "{}", ch)?;
            }
            write!(f, "\n")?;
        }

        Ok(())
    }
}

///
/// https://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html
///
struct IdxData {
    sizes: Vec<usize>,
    data: Vec<u8>,
}

impl IdxData {
    fn load(path: &str) -> IdxData {
        let mut f = File::open(path).unwrap();
        let mut four = [0u8; 4];
        f.read_exact(&mut four).unwrap();
        // The first 2 bytes are always 0.
        assert!(four[0] == 0 && four[1] == 0);
        // 0x08: unsigned byte
        // 0x09: signed byte
        // 0x0B: short (2 bytes)
        // 0x0C: int (4 bytes)
        // 0x0D: float (4 bytes)
        // 0x0E: double (8 bytes)
        assert!(four[2] == 8, "support unsigned only");
        let dim = four[3] as usize;
        let mut sizes = vec![];
        for _ in 0..dim {
            f.read_exact(&mut four).unwrap();
            let n = u32::from_be_bytes(four);
            sizes.push(n);
        }

        let mut data = vec![];
        f.read_to_end(&mut data).unwrap();
        let total = sizes.clone().into_iter().reduce(|acc, e| acc * e).unwrap();
        assert_eq!(data.len(), total as usize);

        IdxData {
            sizes: sizes.into_iter().map(|n| n as usize).collect(),
            data,
        }
    }

    fn dimension(&self) -> usize {
        self.sizes.len()
    }
}

#[test]
fn test_load_image() {
    use crate::common::IMG_RESOLUTION;
    let imgs = ImageSet::loads("data/train-images-idx3-ubyte");
    assert_eq!(imgs.len, 60000);
    let i0 = imgs.iter().next().unwrap();
    assert_eq!(i0.data.len(), IMG_RESOLUTION);
    assert_eq!(i0.width * i0.height, IMG_RESOLUTION);
}

#[test]
fn test_load_labels() {
    let ls = LabelSet::loads("data/train-labels-idx1-ubyte");
    assert_eq!(ls.len(), 60000);
}
