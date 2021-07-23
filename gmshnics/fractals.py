import numpy as np


def koch_snowflake(nsteps, initial=None):
    '''Koch snowflake based on equil. triangle with base [0, 0] -- [1, 0]'''
    assert nsteps > 0

    if initial is None:
        fractal = [np.zeros(2),
                   np.array([np.cos(np.pi/3), np.sin(np.pi/3)]),
                   np.array([1, 0])]
    else:
        assert initial.shape == (3, 2)
        fractal = [intial[0], initial[1], initial[2]]
    # Check equilateral
    _, = set(np.round(np.linalg.norm(fractal[i]-fractal[j]), 8)
             for (i, j) in ((0, 1), (1, 2), (2, 0)))
    
    # Rot.
    R = np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)],
                  [np.sin(np.pi/3), np.cos(np.pi/3)]])

    def koch_rule(A, B, R=R):
        # 3 segments
        X, Y = A + (B-A)/3., A + 2*(B-A)/3.
        # Grow the tent
        Z = X + R.dot(Y-X)

        return [X, Z, Y]
    
    fractal = grow_fractal(fractal, koch_rule, nsteps)

    return fractal


def grow_fractal(fractal, rule, nsteps):
    '''Grow fractal applying the rule to consecutive pairs of points'''
    # Base case
    if nsteps == 0:
        return fractal
    
    l = len(fractal)
    # Preconpute new points for current pairs. Note that we assume that
    # points form a loop
    new_pts = [rule(fractal[i], fractal[(i+1)%l]) for i in range(l)]
    # Insert new points in between segments of the old fractal
    fractal = sum(([p]+pts for p, pts in zip(fractal, new_pts)), [])

    L = len(fractal)
    perimeter = sum(np.linalg.norm(fractal[i]-fractal[(i+1)%L]) for i in range(L))
    print(f'Fractal npoints change {l} to {L}. Perimeter {perimeter}')
    # Next round
    return grow_fractal(fractal, rule, nsteps-1)    
