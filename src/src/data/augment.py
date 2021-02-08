import torch 
import torchio as tio 
import numpy as np 
import random 
import SimpleITK as sitk

from torch import Tensor

spatial_transform = tio.Compose(
        [
            tio.OneOf(
                {
                    tio.RandomAffine(): 0.66, 
                    tio.RandomElasticDeformation(): 0.33,
                },
                p=0.8
            ),
            tio.RandomAnisotropy(p=0.25)
        ]
)

intensity_transform = tio.Compose(
    [
        tio.OneOf(
            {
                tio.RandomNoise(): 0.33, 
                tio.RandomBiasField(): 0.33, 
                tio.RandomGhosting(): 0.33, 
            },
            p=0.50
        ),
        tio.OneOf(
            {
                tio.RandomMotion(): 0.5,
                tio.RandomBlur(): 0.5,
            },
            p=0.50
        )
    ]
)



class RicianNoise(object):
    def __init__(self, noise_level):
        """
        Fourier transformed Gaussian Noise is Rician Noise.
        :param noise_level: The amount of noise to add
        """
        self.noise_level = noise_level

    def add_complex_noise(self, inverse_image, noise_level):
        # Convert the noise from decibels to a linear scale: See:   
        noise_level_linear = 10 ** (noise_level / 10)
        # Real component of the noise: The noise "map" should span the entire image, hence the multiplication
        real_noise = np.sqrt(noise_level_linear / 2) * np.random.randn(inverse_image.shape[0],
                                                                       inverse_image.shape[1], inverse_image.shape[2])
        # Imaginary component of the noise: Note the 1j term
        imaginary_noise = np.sqrt(noise_level_linear / 2) * 1j * np.random.randn(inverse_image.shape[0],
                                                                                 inverse_image.shape[1], inverse_image.shape[2])
        noisy_inverse_image = inverse_image + real_noise + imaginary_noise
        return noisy_inverse_image

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            if len(self.noise_level) == 2:
                noise_level = np.random.randint(self.noise_level[0], self.noise_level[1])
                noise_level = noise_level
            else:
                noise_level = self.noise_level[0]
            
            # Fourier transform the input image
            inverse_image = np.fft.fftn(image)
            # Add complex noise to the image in k-space
            inverse_image_noisy = self.add_complex_noise(inverse_image, noise_level)
            # Reverse Fourier transform the image back into real space
            complex_image_noisy = np.fft.ifftn(inverse_image_noisy)
            # Calculate the magnitude of the image to get something entirely real
            magnitude_image_noisy = np.sqrt(np.real(complex_image_noisy) ** 2 + np.imag(complex_image_noisy) ** 2)
        else:
            magnitude_image_noisy = image
        return magnitude_image_noisy



class ElasticDeformationsBspline(object):
    def __init__(self, num_controlpoints=5, sigma=1):
        """
        Elastic deformations class
        :param num_controlpoints:
        :param sigma:
        """
        self.num_controlpoints = num_controlpoints
        self.sigma = sigma

    def create_elastic_deformation(self, image, num_controlpoints, sigma):
        """
        We need to parameterise our b-spline transform
        The transform will depend on such variables as image size and sigma
        Sigma modulates the strength of the transformation
        The number of control points controls the granularity of our transform
        """
        # Create an instance of a SimpleITK image of the same size as our image
        itkimg = sitk.GetImageFromArray(np.zeros(image.shape))
        # This parameter is just a list with the number of control points per image dimensions
        trans_from_domain_mesh_size = [num_controlpoints] * itkimg.GetDimension()
        # We initialise the transform here: Passing the image size and the control point specifications
        bspline_transformation = sitk.BSplineTransformInitializer(itkimg, trans_from_domain_mesh_size)
        # Isolate the transform parameters: They will be all zero at this stage
        params = np.asarray(bspline_transformation.GetParameters(), dtype=float)
        # Let's initialise the transform by randomly initialising each parameter according to sigma
        params = params + np.random.randn(params.shape[0]) * sigma
        bspline_transformation.SetParameters(tuple(params))
        return bspline_transformation

    def __call__(self, image):
        prob = random.uniform(0, 1)
        if prob > 0.5:
            if len(self.num_controlpoints) == 2:
                num_controlpoints = np.random.randint(self.num_controlpoints[0], self.num_controlpoints[1])
                num_controlpoints = num_controlpoints
            else:
                num_controlpoints = self.num_controlpoints[0]
            if len(self.sigma) == 2:
                sigma = np.random.uniform(self.sigma[0], self.sigma[1])
                sigma = sigma
            else:
                sigma = self.sigma[0]
            # We need to choose an interpolation method for our transformed image, let's just go with b-spline
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkBSpline)
            # Let's convert our image to an sitk image
            sitk_image = sitk.GetImageFromArray(image)
            # sitk_grid = self.create_grid(image)
            # Specify the image to be transformed: This is the reference image
            resampler.SetReferenceImage(sitk_image)
            resampler.SetDefaultPixelValue(0)
            # Initialise the transform
            bspline_transform = self.create_elastic_deformation(image, num_controlpoints, sigma)
            # Set the transform in the initialiser
            resampler.SetTransform(bspline_transform)
            # Carry out the resampling according to the transform and the resampling method
            out_img_sitk = resampler.Execute(sitk_image)
            # out_grid_sitk = resampler.Execute(sitk_grid)
            # Convert the image back into a python array
            out_img = sitk.GetArrayFromImage(out_img_sitk)
            out_img = out_img.reshape(image.shape)
        else:
            out_img = image
        return out_img