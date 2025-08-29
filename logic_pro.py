import cv2
import numpy as np
import tensorflow as tf


try:
    model = tf.keras.models.load_model('handwritten_pro_model.keras') # Load the pre-trained CNN model
except Exception as e:   # Handle model loading errors
    print(f"Error loading model: {e}")
    exit()  # Exit if model cannot be loaded

cap = cv2.VideoCapture(0)  # Start video capture from the default camera, 0 indicates the default camera
if not cap.isOpened(): # Check if the camera opened successfully
    print("Error: Could not open camera.")
    exit()

#main loop
while True:
    ret, frame = cap.read() # Capture frame-by-frame from the camera. ret is a boolean indicating if the frame was read correctly and frame is the image itself.
    #cap.read()returns two values: a boolean (ret) and the frame itself (frame).

    if not ret: # If frame reading was not successful, break the loop
        break

    frame = cv2.flip(frame, 1) # Flip the frame horizontally for a mirror effect 

    # --- Part 3: Region of Interest (ROI) ---
    roi_x1, roi_y1 = 100, 100   # Define the top-left corner of the ROI(Region of Interest). This is where the user should place their hand-written digit.
                                #100 pixels from the left and 100 pixels from the top of the frame.

    roi_x2, roi_y2 = 300, 300    # Define the bottom-right corner of the ROI. This is 300 pixels to the right and 300 pixels down from the top-left corner.
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2) # Draw a green rectangle around the ROI on the frame for visual guidance.  
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # --- Part 4: Image Processing Pipeline ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) # Convert the ROI to grayscale. BGR(Blue, Green, Red) is the default color format in OpenCV.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Apply Gaussian blur to the grayscale image to reduce noise and improve contour detection.
    _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV) # Apply binary inverse thresholding to create a binary image. Pixels above 120 become 0 (black), and those below become 255 (white).
    # Find contours in the thresholded image. Contours are simply the boundaries of white regions in the binary image.

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Retrieve only the external contours and compress horizontal, vertical, and diagonal segments.
                                                                                        #(contours, _ ) the _ is used to ignore the second value given by the function as we do not require it.
    # --- Part 5: Prediction Logic ---
    prediction_text = "Prediction: None"
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 50:  #"Proceed only if (1) we found at least one shape AND (2) the biggest shape we found has an area of more than 50 pixels."
        largest_contour = max(contours, key=cv2.contourArea)   # Find the largest contour by area.
        x, y, w, h = cv2.boundingRect(largest_contour)  # Get the bounding box coordinates for the largest contour.
        
        digit = thresh[y:y+h, x:x+w]  # Extract the digit from the thresholded image using the bounding box coordinates.
        digit = cv2.flip(digit, 1)    # Flip the digit horizontally to correct orientation for the model
        padded_digit = cv2.copyMakeBorder(digit, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0]) # Add a 10-pixel black border around the digit to ensure it fits well in the 28x28 input size.
        resized_digit = cv2.resize(padded_digit, (28, 28))  # Resize the image to 28x28 pixels, which is the input size expected by the CNN model.
        
        prepared_digit = resized_digit.reshape(1, 28, 28, 1)  # Reshape the image to match the input shape of the model: (1, 28, 28, 1). The first dimension is the batch size (1 image), and the last dimension is the color channel (1 for grayscale).
        prepared_digit = prepared_digit / 255.0 # Normalize pixel values to be between 0 and 1 by dividing by 255.0.

        prediction = model.predict(prepared_digit) # Use the CNN model to predict the digit.
        predicted_digit = np.argmax(prediction)  # Get the index of the highest probability from the model's output, which corresponds to the predicted digit (0-9).
        prediction_text = f"Prediction: {predicted_digit}"  # Prepare the prediction text to display on the frame.

    # --- Part 6: Display Results ---
    cv2.putText(frame, prediction_text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)  # Display the prediction text on the frame at position (10, 30) using a red font.
    #cv2.imshow('Thresholded ROI', thresh)  # Show the thresholded ROI(image which is seen by the model) in a separate window for debugging purposes.
    #cv2.imshow('Live Digit Recognition', frame)  # Show the main frame with the ROI and prediction in a window.
    
# Flip the threshold image horizontally for a more intuitive debug view
    display_thresh = cv2.flip(thresh, 1)

# --- And change this line to use the new flipped image ---
    cv2.imshow('Thresholded ROI', display_thresh) 
    cv2.imshow('Live Digit Recognition', frame)

    # --- Part 7: Quit Logic ---
    if cv2.waitKey(1) & 0xFF == ord('q'): # "Give the display windows a moment to update, and during that moment, check if the user has pressed the 'q' key. If they have, stop the program."
        break

# --- Part 8: Cleanup ---
cap.release() #properly releases the webcam.
cv2.destroyAllWindows() #closes all OpenCV windows to free up resources.