# Image-Local-Feature-Matching
*An interesting Computer Vision project - generating image features around local points within the image.*

Here's the outline what this project comprises of in `student.py`:

1. Generating Interest Points with Harris `get_interest_points`
2. Creating SIFT-like features around each identified interest point via `get_features`
3. Matching Features between two images using `match_features`

## A glimpse of the Results:

### Notre Dame
![](images/notre_dame_matches.jpg)
<br />
*Matches after the complete pipeline execution on Notre Dame. Matches: 1113, Ac