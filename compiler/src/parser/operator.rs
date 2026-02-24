pub enum Functions {
    // Mappings
    Identity,
    Sparse(f64),
    Patch(u32,u32,Option<u32>),
    Stride(u32,u32,u32,u32),
    Broadcast,

    // Aggregations
    Pool,
    Concat,
    Spatial(u32,u32),
    Vote,

    // Activation / output transforms
    WeightedSum,
    Argmax,
    Softmax,
    Threshold(f64),

    // Statistics functions
    Sum,
    Mean,
    Max,
    Min,
    Std,
    Var,
    Abs,
    Sign,
    Round,
    Floor,
    Ceil,
    Trunc,
    Exp,
    Log,
    Sin,
    Cos,

    // gating functions
    Gated(f64),

    // learning functions
    Hebbian(f64),
    AntiHebbian(f64),
    STDP(f64),
    Covariance(f64),
    Oja(f64),
    Delta(f64),
    Perceptron(f64),
    Adagrad(f64),
    Adam(f64),
    RMSprop(f64),
    SGD(f64),
    Momentum(f64),
    Nesterov(f64),
    AdaDelta(f64),
    AdaGrad(f64),
    Adamax(f64),
    Nadam(f64),
}


pub enum Operators {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulus,
    Absolute,
    Sign,
    Round,
    Floor,
    Ceil,
    Trunc,
    Not,
    And,
    Or,
    Xor,
    NotEqual,
    Equal,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
}