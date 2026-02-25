struct Tensor {
    dimensions: Vec<u32>,
    init: TensorInit,
}

enum TensorInit {
    Zeros,
    Ones,
    RandUniform,
    RandNormal,
}