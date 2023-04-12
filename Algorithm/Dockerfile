# Use a pre-built PyTorch image as the base
FROM python:3

# Set the working directory to /app
WORKDIR /app

# Install any additional dependencies
RUN pip3 install mediapipe==0.9.0.1

# Copy the inference code to the container
COPY pose_estimation.py /app/pose_estimation.py
COPY utils.py /app/utils.py
COPY ./test/vid_6.mp4 app/test/vid_6.mp4


# Define the command to run when the container starts
CMD ["python", "pose_estimation.py"]
