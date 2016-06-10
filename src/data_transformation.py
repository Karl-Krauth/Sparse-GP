import numpy as np
from sklearn import preprocessing


class DataTransformation(object):
    """
    A generic class for the transformation of data
    """
    def __init__(self, X, Y):
        pass

    def transform_X(self, X):
        """
        transforms X

        :param
         X: Input X
        :return
         transformed X
        """
        raise NotImplementedError()

    def transform_Y(self, Y):
        """
        transforms Y

        :param
         Y: Input Y
        :return
         transformed Y
        """
        raise NotImplementedError()

    def untransform_X(self, X):
        """
        Untransforms X to its original values

        :param
         X: transformed X
        :return
         untransformed X
        """
        raise NotImplementedError()

    def untransform_Y(self, Y):
        """
        Untransforms Y
        :param
         Y: transformed Y
        :return
         untransfomred Y
        """
        raise NotImplementedError()

    def untransform_Y_var(self, Yvar):
        raise NotImplementedError()

    def untransform_NLPD(self, NLPD):
        """
        Untransfomrs NLPD to the original Y space

        :param
         NLPD: transfomred NLPD
        :return
         untransformed NLPD
        """
        raise NotImplementedError()


class IdentityTransformation(DataTransformation):
    """
    Identity transformation. No transformation will be applied to data.
    """

    def __init__(self, X ,Y):
        pass

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return Y

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return Y

    def untransform_Y_var(self, Yvar):
        return Yvar

    def untransform_NLPD(self, NLPD):
        return NLPD


class MeanTransformation(DataTransformation):
    """
    Only transforms Y as follows:
    transformed Y = untransformed Y - mean(Y)
    """

    def __init__(self, X, Y):
        self.mean = Y.mean(axis=0)

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return Y - self.mean

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return Y + self.mean

    def untransform_Y_var(self, Yvar):
        return Yvar

    def untransform_NLPD(self, NLPD):
        return NLPD


class MeanStdYTransformation(DataTransformation):
    """
    Transforms only Y in a way that the transformed Y has mean = 0 and std =1
    """

    def __init__(self, X, Y):
        self.scaler = preprocessing.StandardScaler().fit(Y)

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return self.scaler.transform(Y)

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return self.scaler.inverse_transform(Y)

    def untransform_Y_var(self, Yvar):
        return Yvar

    def untransform_NLPD(self, NLPD):
        return NLPD + np.hstack((np.array([np.log(self.scaler.std_).sum()]), np.log(self.scaler.std_)))


class MeanStdTransformation(DataTransformation):
    def __init__(self, X, Y):
        self.input_scaler = preprocessing.StandardScaler().fit(X)
        self.output_scaler = preprocessing.StandardScaler().fit(Y)

    def transform_X(self, X):
        return self.input_scaler.transform(X)

    def transform_Y(self, Y):
        return self.output_scaler.transform(Y)

    def untransform_X(self, X):
        return self.input_scaler.inverse_transform(X)

    def untransform_Y(self, Y):
        return self.output_scaler.inverse_transform(Y)

    def untransform_Y_var(self, Yvar):
        return Yvar

    def untransform_NLPD(self, NLPD):
        return NLPD + np.hstack([np.array([np.log(self.output_scaler.std_).sum()]),
                                 np.log(self.output_scaler.std_)])


class MinTransformation(DataTransformation):
    """
    Transforms only Y.
    transformed Y = (Y - min(Y)) / (max(Y) - min(Y)) - 0.5
    """

    def __init__(self, X, Y):
        self.min = Y.min()
        self.max = Y.max()
        self.offset = 0.5

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return (Y-self.min).astype('float')/(self.max-self.min) - self.offset

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return (Y+self.offset)*(self.max-self.min) + self.min

    def untransform_Y_var(self, Yvar):
        return Yvar * (self.max-self.min) ** 2

    def untransform_NLPD(self, NLPD):
        return NLPD + np.log(self.max - self.min)
