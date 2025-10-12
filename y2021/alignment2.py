
import sys
import itk
import argparse

from distutils.version import StrictVersion as VS

if VS(itk.Version.GetITKVersion()) < VS("4.9.0"):
    print("ITK 4.9.0 is required.")
    sys.exit(1)



PixelType = itk.ctype("float")

fixedImage = itk.imread(r"Images/HXSheets/hxsheet5.png", PixelType)
movingImage = itk.imread(r"Images/HXSheets/hxsheet4.png", PixelType)

Dimension = fixedImage.GetImageDimension()
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]

TransformType = itk.TranslationTransform[itk.D, Dimension]
initialTransform = TransformType.New()

optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
    LearningRate=4,
    MinimumStepLength=0.001,
    RelaxationFactor=0.5,
    NumberOfIterations=200,
)

metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()

registration = itk.ImageRegistrationMethodv4.New(
    FixedImage=fixedImage,
    MovingImage=movingImage,
    Metric=metric,
    Optimizer=optimizer,
    InitialTransform=initialTransform,
)

movingInitialTransform = TransformType.New()
initialParameters = movingInitialTransform.GetParameters()
initialParameters[0] = 0
initialParameters[1] = 0
movingInitialTransform.SetParameters(initialParameters)
registration.SetMovingInitialTransform(movingInitialTransform)

identityTransform = TransformType.New()
identityTransform.SetIdentity()
registration.SetFixedInitialTransform(identityTransform)

registration.SetNumberOfLevels(1)
registration.SetSmoothingSigmasPerLevel([0])
registration.SetShrinkFactorsPerLevel([1])

registration.Update()

transform = registration.GetTransform()
finalParameters = transform.GetParameters()
translationAlongX = finalParameters.GetElement(0)
translationAlongY = finalParameters.GetElement(1)

numberOfIterations = optimizer.GetCurrentIteration()

bestValue = optimizer.GetValue()

print("Result = ")
print(" Translation X = " + str(translationAlongX))
print(" Translation Y = " + str(translationAlongY))
print(" Iterations    = " + str(numberOfIterations))
print(" Metric value  = " + str(bestValue))

CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
outputCompositeTransform = CompositeTransformType.New()
outputCompositeTransform.AddTransform(movingInitialTransform)
outputCompositeTransform.AddTransform(registration.GetModifiableTransform())

resampler = itk.ResampleImageFilter.New(
    Input=movingImage,
    Transform=outputCompositeTransform,
    UseReferenceImage=True,
    ReferenceImage=fixedImage,
)
resampler.SetDefaultPixelValue(100)

OutputPixelType = itk.ctype("unsigned char")
OutputImageType = itk.Image[OutputPixelType, Dimension]

caster = itk.CastImageFilter[FixedImageType, OutputImageType].New(Input=resampler)

writer = itk.ImageFileWriter.New(Input=caster, FileName="AlignOut.png")
writer.Update()

difference = itk.SubtractImageFilter.New(Input1=fixedImage, Input2=resampler)

intensityRescaler = itk.RescaleIntensityImageFilter[
    FixedImageType, OutputImageType
].New(
    Input=difference,
    OutputMinimum=itk.NumericTraits[OutputPixelType].min(),
    OutputMaximum=itk.NumericTraits[OutputPixelType].max(),
)

resampler.SetDefaultPixelValue(1)
writer.SetInput(intensityRescaler.GetOutput())
writer.SetFileName("difference_image_after.png")
writer.Update()

resampler.SetTransform(identityTransform)
writer.SetFileName("difference_image_before.png")
writer.Update()