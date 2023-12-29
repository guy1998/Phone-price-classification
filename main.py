from Algorithms.feed_forward_neural_network import mlp_model

if __name__ == "__main__":
    mlp_model("fe_screen_size", "z_score", 0.8, 20, 'adam',
              activation='identity')
