use idx::{ImageSet, LabelSet};
use network::Network;

mod common;
mod idx;
mod network;

fn main() {
    let mut nw = Network::new();
    let train_images = ImageSet::loads("data/train-images.idx3-ubyte");
    let train_labels = LabelSet::loads("data/train-labels.idx1-ubyte");
    // for (i, l) in train_images.iter().zip(train_labels.iter()) {
    //     println!("{}\n{}", i, l);
    //     break;
    // }
    nw.train(&train_images, &train_labels, 10);

    let exam_imgs = ImageSet::loads("data/t10k-images.idx3-ubyte");
    let exam_labels = LabelSet::loads("data/t10k-labels.idx1-ubyte");
    nw.predict(&exam_imgs, &exam_labels);
}
