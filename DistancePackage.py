from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import numpy as np

class Visualize:
  """
  Visualization class
  """
  
  def PlotVec(self, v, show=False, c='k'):
    """
    Plot a 2D vector

    v : 2 dim vector
    show : [False] boolean, plot vector
    c : ['k'] character or [f, f, f], color of vector
    """
    plt.plot([0, v[0]], [0, v[1]], color=c)
    if show:
      plt.show()


class DistanceLearn:
  """
  Distance Learn Algorithm:
  
  Given a matrix X where the rows are samples and the columns are features and pairwise similarity between samples, 
  create a new space that preserves pairwise distances between samples in feature space. In this transformed space,
  the Euclidean distance between points is proportional to the error between points.
  """
  
  def CalcAngle(self, v, u):
    return np.arccos( np.dot(v,u)/(np.linalg.norm(v)*np.linalg.norm(u)))
  
  def RotateVec( self, u, angle ):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    if np.shape( u ) != [2,1]:
      np.reshape(u, (2,1))
    return np.dot(R, u)
  
  
  def InitializeTransformMatrix(self):
    """
    InitializeTransformMatrix:
    
    Initialize transformation matrix so that angle between vectors (stimuli) function of similarity. This 
    space is meaningless at this stage - will need to rotate space s.t. transform is invariant to ones vector 
    """
    S = np.eye(self.num_samples)
    a = (np.pi/2 - np.arccos(self.confusion_matrix[0,1]))/2
    print(a)
    S[:2,0] = self.RotateVec(S[:2,0], a)
    S[:2,1] = self.RotateVec(S[:2,1], -a)
    for i in range(2,self.num_samples):
      d = []
      for j in range(i):
        d.append( self.confusion_matrix[i,j] )
      S[:,i] = np.dot( np.array(d), np.linalg.pinv(S[:,range(i)]) )
      l = np.square(np.linalg.norm(S[:,i]))
      if (l <= 1):
        S[i,i] = np.sqrt(1 - np.square(np.linalg.norm(S[:,i])))
      else:
        S[i,i] = -np.sqrt(np.square(np.linalg.norm(S[:,i]))-1)
    return S

  def InitializeWeights(self):
    # initialize with identity matrix
    return np.reshape(np.eye(self.num_samples), (self.num_samples*self.num_samples))
  
  def ObjFunc(self, T):
    """
    Define objective function for rotating. Specifically want to rotate such that transform is invariant to 
    ones vector.
    """
    Ts = []
    As = []
    for i in range(self.num_features):
      Ts.append(T[i*self.num_samples:(i+1)*self.num_samples])
      As.append(np.sum(self.A[i,:]))
    
    As = np.array(As)
    o = 0
    for t in Ts:
      o = o + (0.5*(1-np.dot(t,As))**2)
    return o
  
  def ObjFuncDeriv(self, T):
    """
    Derivative of objective function.
    """
    dfdT = []
    Ts = []
    As = []
    for i in range(self.num_samples):
      Ts.append(T[i*self.num_samples:(i+1)*self.num_samples])
      As.append(np.sum(self.A[i,:]))
    
    As = np.array(As)
    for i in range(self.num_samples):
      for j in range(i*self.num_samples, (i+1)*self.num_samples):
        dfdT.append( -(1-np.dot(Ts[i],As)) * (T[j]*As[j%self.num_samples]) )
    return np.array(dfdT)

  def GenLambdaCons1(self, Tp, i, epsilon):
    """
    First constraint for objective function: 
    
    rotate s.t. norm of axes of compressed space is fixed.
    """
    return lambda t: -0.5*np.linalg.norm(self.A[:,i] - np.dot(Tp, self.A[:,i]))**2 + epsilon
  
  def GenLambdaCons2(self, Tp, i, j, epsilon):
    """
    Second constraint for objective function:
    
    rotate s.t. angle between vectors is fixed (reduces to making sure dot product is identical because
    of constraint 1)
    """
    TA1 = np.dot(Tp, self.A[:,i])
    TA2 = np.dot(Tp, self.A[:,j])
    return lambda t: -0.5*(np.dot(self.A[:,i], self.A[:,j]) - np.dot(TA1, TA2))**2 + epsilon
  
  def GenConstraints(self, T, epsilon):
    """
    Generate all them constraints
    """
    Tp = np.reshape(T, (self.num_samples, self.num_samples) )
    C = []
    for i in range(self.num_samples):
      C.append(self.GenLambdaCons1(Tp, i,epsilon))
    
    for i in range(self.num_samples):
      for j in range(i, self.num_samples):
        C.append(self.GenLambdaCons2(Tp, i, j, epsilon))
            
    cons=[]
    for c in C:
      cons.append({'type': 'ineq',
                   'fun' : c})
    return cons
    
  def GetTransformMatrix(self):
    return self.transform_matrix
  
  def LearnSpace(self, X, confusion_matrix, epsilon=0):
    """
    LearnSpace:
    
    Transforms a matrix of samples and features such that samples that are similar to each other are closer 
    together, while samples that are different are further from each other. This is a transformation of the 
    feature space in which the stimuli exist.
    
    Input:
    X: (num_samples X num_features), a matrix where the columns are the features and the rows are samples
    confusion_matrix: (num_samples X num_samples), a similarity matrix where the (i,j)th entry is the similarity between samples i and j
    epsilon: [default 0], a tradeoff parameter. Larger epsilon means preserving angle is more important, smaller epsilon means invariance to ones vector most important

    """
    self.confusion_matrix = confusion_matrix
    self.epsilon = epsilon
    self.F = np.copy(X.T)
    self.X = np.copy(X)
    self.num_features = np.shape(self.F)[0]
    self.num_samples = np.shape(self.X)[0]
    self.A = self.InitializeTransformMatrix()
    self.T = self.InitializeWeights()
    self.constraints = self.GenConstraints( self.T, self.epsilon )
    self.res = minimize(self.ObjFunc, self.T, jac=self.ObjFuncDeriv, 
                        constraints=self.constraints, method='SLSQP', options={'disp':True} )
    self.T = np.reshape(self.res.x, (self.num_samples, self.num_samples))
    self.transform_matrix = np.dot(self.T,self.A)
    self.transformed_X = np.dot(self.transform_matrix, self.X)
    self.transformed_F = np.copy(self.transformed_X.T)
    
    print("1s vector returns: ")
    print(np.dot(self.transform_matrix, np.ones(self.num_samples)))
    print("Transformed input: ")
    print(self.transformed_X)

    return self.transformed_X
          
  def LearnRegression(self, model_type='krr', kernel='rbf', epsilon=.1, degree=3, validate=True):
    """
    LearnRegression:

    Learn the regression to map new points into the transformed space.

    Input:
    model_type : ['krr'] regression algorithm: 'ols', 'nn', 'krr'
    kernel : ['rbf'] type of kernel to use: 'linear', 'poly', 'rbf'
    """
    self.model_type=model_type
    self.mean = np.mean(self.X, axis=0)
    self.std = np.std(self.X, axis=0)
    Xp = (self.X - self.mean)/self.std
    Xp = np.hstack( (Xp, np.ones((np.shape(Xp)[0],1)) ))
    if (model_type == 'krr'):
      print('Training KRR')
      self.model = KernelRidge(kernel=kernel)
      self.model.fit(Xp, self.transformed_X)
      print('Model successfully fit')
    
    if validate:
      error = self.Validate( self.X, self.transformed_X)
      print("Empirical error: %f")%error


  
  def Predict(self, X):
    """
    Predict:

    Predict y_hat for model that was previously learned.
    
    Input:
    X : Matrix where rows are samples and columns are features
    """
    print('Predicting values')
    N = np.shape(X)[0]

    Xp = (self.X - self.mean)/self.std
    Xp = np.hstack( ( Xp, np.ones((np.shape(Xp)[0],1)) ) )
    if (self.model_type == 'krr'):
      Y_hat = self.model.predict(Xp)
    return Y_hat

  def Validate(self, X, Y):
    """
    Validate:

    Validate the regression model
    """
    def MultStd(v):
      return v*self.std

    def AddMean(v):
      return v*self.mean

    N = np.shape(X)[0]
    Y_hat = self.Predict(X)

    error = 0
    for i in range(N):
      error = error + np.linalg.norm(Y[i] - Y_hat[i])

    return error
