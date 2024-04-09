# Some Handcoded Policies for Balancing a Pole on a Cart

I show two examples:

1. Pole is slightly out of balance, but without any angular speed
2. Pole is slightly out of balance, with some angular speed

## Noop policy
Don't apply any force to the cart.

<img src="gif/move_nothing.gif">
<img src="gif/move_nothing2.gif">

## Constant force policy
This policy applies a constant force to the cart.

<img src="gif/move_constant.gif">
<img src="gif/move_constant2.gif">

## Random force policy
This policy applies a random force to the cart.

<img src="gif/move_random.gif">
<img src="gif/move_random2.gif">

## Move opposite policy
This policy applies a force based on the pole's angle. If it's "on the
right", apply a force to the right, and vice versa.

<img src="gif/move_opposite.gif">
<img src="gif/move_opposite2.gif">

## Move opposite (improved) policy
Similar to the previous one, but try to "swing the pole up" when it's in the
lower half.

<img src="gif/move_opposite_upswing.gif">
<img src="gif/move_opposite_upswing2.gif">
