# thu-web-IR-project-api

To run the server:

```
flask --app app run
```

## Deployment
Start by running the following commands

```
pip install build

pip install waitress

python -m build --wheel

waitress-serve --host=0.0.0.0 --port=8000 app:app
```

For macOS, run the following command to find you IP-adress:
```ipconfig getifaddr en0```

http://{IP-address}:8000 is the address others can access.