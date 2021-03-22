BUCKET=gs://vaex-data

# Make new package
python setup.py sdist --formats=gztar

# Copy it to GCS
gsutil cp dist/*tar.gz $BUCKET/training-modules/

echo "Clean up..."
rm -r dist
rm -r har_model.egg-info

echo "Created a training module and copied it to the specified GCS location."