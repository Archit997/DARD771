import numpy as np
import sklearn.linear_model as lm
import sklearn.svm as svm
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################
	X_train = my_map(X_train)
	model = lm.LogisticRegression()
	#model = svm.LinearSVC(loss='hinge')
	model.fit(X_train,y_train)
	w = model.coef_.flatten()
	b = model.intercept_


	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challeenge bits
	# y_train contains the responses
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
	return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################
	X = 1-2*X
	X_rev = X[:,::-1]
	X_prod = np.cumprod(X_rev,axis=1)
	X = X_prod[:,::-1]

	n_samples, n_features = X.shape
	outer_product = np.einsum('ij,ik->ijk', X, X)
	upper_triangle_indices = np.triu_indices(n_features,k=1)
	features = outer_product[:, upper_triangle_indices[0], upper_triangle_indices[1]]
	feat = np.concatenate((X,features),axis=1)

	return feat
