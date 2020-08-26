from flask import Flask, request, send_from_directory
from predict import predict

app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict_svr():
    adj_path = request.args.get('adj', type=str)
    fea_path = request.args.get('feature', type=str)

    # Run the predict module, gen the file.
    # res_pkl_path: str, the path of res pkl.
    res_pkl_name = predict(adj_path, fea_path)
    return send_from_directory('/home/jtli/Files/Competition_KDD/', res_pkl_name, as_attachment=True)


if __name__ == '__main__':
    app.run(port=5000)
