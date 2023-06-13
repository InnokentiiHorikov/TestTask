def compiling_model(model):
  model.compile(optimizer=Adam(1e-3, decay=1e-3), loss=DICE_score_loss, metrics=["binary_accuracy", DICE_score])

  model.fit(train_x, train_y,
          batch_size = 32,
                    epochs = 200)


if __name__ == '__main__':
  model_inference()
