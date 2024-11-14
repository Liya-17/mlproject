# classifier/views.py
from django.shortcuts import render
from django.conf import settings
from .load_model import classify_image
import os

def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        # Save uploaded image to a temporary path
        uploaded_image = request.FILES["image"]
        img_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)

        with open(img_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Classify the uploaded image and get predictions
        predictions = classify_image(img_path, top_n=3, confidence_threshold=0.5)

        # Prepare the results to send to the template
        results = [
            {"description": description, "score": f"{score * 100:.2f}%"}
            for _, description, score in predictions
        ]

        # Render the results page with predictions
        return render(request, "classifier/result.html", {"results": results})

    # If GET request or no image uploaded, render the upload page
    return render(request, "classifier/upload.html")
