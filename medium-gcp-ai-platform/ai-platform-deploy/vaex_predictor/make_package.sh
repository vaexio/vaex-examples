BUCKET=gs://vaex-data

# Remove file
if [ -f dist ] ; then
    rm -r dist
    rm vaex_predictor.egg-info
fi

# Make new package
python setup.py sdist --formats=gztar

# Copy it to GCS
gsutil cp dist/vaex_predictor-0.0.0.tar.gz $BUCKET/deployments/
