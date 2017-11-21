This repository contains the server code for my `Deep Learning Recommender Systems` talk.  As a disclaimer, it is likely not set up well to run without changes being made, and is not incredibly well-designed.  That said, it works, and the code may still be helpful in understanding how the models were deployed behind a web server.

The project is based on data from [MovieLens](https://grouplens.org/datasets/movielens/), code from [Robin Devooght](https://github.com/rdevooght/sequence-based-recommendations), and these two papers:

Devooght, Robin and Hugues Bersini "Collaborative Filtering with Recurrent Neural Networks" (2017)

[https://arxiv.org/abs/1608.07400](https://arxiv.org/abs/1608.07400)


Kuchaiev, Oleksii and Boris Ginsburg "Training Deep Autoencoders for Collaborative Filtering" (2017)

[https://arxiv.org/abs/1708.01715](https://arxiv.org/abs/1708.01715)

The code for implementing the collaborative filtering autoencoder in Keras is in the IPython notebook.

To run the server:

     $ cd webapp
     $ FLASK_APP=app.py flask run

✨🍰✨
