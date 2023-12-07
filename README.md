
# Detection of Face Features using Adapted triplet loss

We worked on detecting and recognising the face features using custom collected dataset. We have used PyTorch as deep learning Framework and have used ResNet weights to finetune the model with five classes in our dataset.

### Live inference can be found at [huggingface](https://huggingface.co/spaces/prudhvirajboddu/detectionmodel)


## Running the project

We have used python 3.11 Version to run this game on and latest pytorch environment


First run this command in a virtual environment to install the dependencies

```bash
  pip install -r requirements.txt
```

After setting up the environment, Run 

```bash
  python train.py
```

This will start training the model and save it to outputs directory.


Once the model is trained, you can do the inference or deploy it to the gradio app. I have written code for deploying it through gradio and inference the model with input. After training the model run

```bash
  python webapp.py
```


