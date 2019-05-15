# AI-Digit_Classification-SVD
0-9 digit classification for Artificial Intelligence using SVD

#An SVD basis classification algorithm
####Training.
For the training set of known digits, compute the SVD of each
class of digits, and use k basis vectors for each class.
####Classification.
For a given test digit, compute its relative residual in all
ten bases. If one residual is significantly smaller than all the others,
classify as that. Otherwise give up.
