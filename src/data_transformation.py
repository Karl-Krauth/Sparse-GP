import numpy as np
from sklearn import preprocessing


class DataTransformation:
    """
    A generic class for the transformation of data
    """

    def __init__(self):
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


class IdentityTransformation:
    """
    Identity transformation. No transformation will be applied to data.
    """

    def __init__(self):
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

    @staticmethod
    def get_transformation(Y, X):
        return IdentityTransformation()

    def untransform_NLPD(self, NLPD):
        return NLPD


class MeanTransformation(object, DataTransformation):
    """
    Only transforms Y as follows:
    transformed Y = untransformed Y - mean(Y)
    """

    def __init__(self, mean):
        super(MeanTransformation, self).__init__()
        self.mean = mean

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

    @staticmethod
    def get_transformation(Y, X):
        return MeanTransformation(Y.mean(axis=0))


class MeanStdYTransformation(object, DataTransformation):
    """
    Transforms only Y in a way that the transformed Y has mean = 0 and std =1
    """

    def __init__(self, scalar):
        super(MeanStdYTransformation, self).__init__()
        self.scalar = scalar

    def transform_X(self, X):
        return X

    def transform_Y(self, Y):
        return self.scalar.transform(Y)

    def untransform_X(self, X):
        return X

    def untransform_Y(self, Y):
        return self.scalar.inverse_transform(Y)

    def untransform_Y_var(self, Yvar):
        return Yvar

    def untransform_NLPD(self, NLPD):
        return NLPD + np.hstack((np.array([np.log(self.scalar.std_).sum()]), np.log(self.scalar.std_)))

    @staticmethod
    def get_transformation(Y, X):
        return MeanStdYTransformation(preprocessing.StandardScaler().fit(Y))


class MinTransformation(object, DataTransformation):
    """
    Transforms only Y.
    transformed Y = (Y - min(Y)) / (max(Y) - min(Y)) - 0.5
    """

    def __init__(self, min, max, offset):
        super(MinTransformation, self).__init__()
        self.min = min
        self.max = max
        self.offset = offset

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

    @staticmethod
    def get_transformation(Y, X):
        return MinTransformation(Y.min(), Y.max(), 0.5)
