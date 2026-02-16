Domains are the objects that allow the network to communicate (receive or ouput information) with the external world.
It is an important question of how to map to neuron_id, activity for arbitrary domain types. For now, I shall inquire as to how to do vision and language.

In general, you want to figure out what maps to activity, and what maps to neuron ids. For images, the pixel values in (0,1) are what you want to map to activity, and it should be additive with decay.

For example, if pixel x is 0.85 in one frame, this activates input sensory neuron with id y with activity = 0.85. This value decays according to some constant the if another frame is shown at time t with x = 0.67, then we do \sigma(x + a(t,x_0))

## MNIST
The MNIST dataset comprises a set of 2-dimensional images with pixel values in (0,1). So we have a pixel position, pixel value set that cleanly maps -> neuron id, activity value without much thought.

## RGB images
Raw space: 3D image tensor `(H, W, C)` where C is a 3 dimensional tensor of values in (0,1). How should we handle a 3 dimensional mapping of (0,1)? The trick is to have a sensory neuron per dimension, so that each pixel maps to at least 3 neurons.

## Language
Language is somewhat tricky. Certainly we want the same set of neurons to activate for the same character.

It seems standard to define the raw space as a token sequence and define a map between token identity -> neuron id. Since there is no analogous continuous tensor associated with the value of the characters, the map could simply activate corresponding neuron values with activity = 1. These values decay.


# Interface Between Domain and Regions
So how can we define an API between arbitrary tensors and neurons?

## Input
1. Clarify what is the tensor to be mapped to activities (if exists).
2. Normalize it to (0,1)^(n)
3. Clarify the size of the discrete set which is to be mapped to neuron_ids (let this be cardinality k).
3. Surject it to a set of (possibly overlapping) neurons of size k \times m \times n across different regions (optional)
where m is the number of neurons that receive a given item in the set.
4. If no continuous activity tensor exists, then map the discrete terms to have activity a = 1 on their respective neuron_ids.

```python
class Domain:
    activity_tensor_shape
    discrete_set_cardinality
    value_range
```

So this provides all the information about the domain, and encapsulates the normalization procedure, making everything ready to map onto neurons. There must be a separate class or function that sets up the sensory regions and assigns ids for given elements of the domain.

## Output
If outputs are to be tensors or classify objects, you could still use the domain class.
The activity tensor shape is how the shape of the continuous output data from a set of neuron activities.
The discrete set cardinality is the number of continuous elements (or number of classifier elements in the case of 0 or None activity_tensor_shape).
The value range is the is the range of the activity tensors for the inverse normalization.

It is necessary to define an order of the discrete set. Instead of discrete set elements from an input domain to activity 1, it makes sense to apply an argmax function to the F_z of a region and take the set element as the corresponding max neuron_id.