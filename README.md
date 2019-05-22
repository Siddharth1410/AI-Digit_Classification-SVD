# AI-Digit_Classification-SVD
0-9 digit classification for Artificial Intelligence using SVD

## An SVD basis classification algorithm
(a) Training samples from each of c classes of objects are chosen, with each object represented
as a vector in R </br>
n where n is the number of attributes describing each object (for example,
pixels for images). </br>
(b) For each class, create an n × q matrix where q is the number of training samples for that
class. </br>
(c) Find the SVD of each of these training matrices.</br>
(d) Given a test sample z (which needs to be represented as a column vector), for each class
calculate
### Training.
#### digitsTrainCombined.csv
this is a 256 × 4000 matrix. Each column represents a 16 × 16 grayscale image of a digit. The first 400 columns are examples of zeros, the
next 400 columns are examples of ones, and so forth, with the last 400 columns being
examples of nines.

### Classification.
#### digitsTestCombined.csv
this is a 256×1000 matrix. Each column represents a 16×16 grayscale image of a digit. The first 100 columns are examples of zeros, the next 100 columns are examples of ones, and so forth, with the last 100 columns being examples of nines.
