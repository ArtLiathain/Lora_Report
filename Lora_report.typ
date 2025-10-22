#set math.equation(numbering: "1.")

= Fine-Tuning Pretrained Large Language Models 


== Training LLMs

Training for an LLM is a long and compute intensive task. 
The level of current models such as GPT5 with 1.8trillion. This initial training
create a generalist model that struggles with specific information and
queries. A more specifically trained model is needed to allow for
accurate domain specific responses. Recreating the model for each task
in this case is highly inefficient and would lead to bloated systems
with many AIs. As every task would require its own fully trained model.
#figure(
  image("images/2025-10-22-12:39:20.png"),
  caption : [One model for every task]
)

= Adapter Architecture

This deficiency in base models opened the door for the adapter
architecture to take place. This approach applies an additional layer to
the AI adapter. This adapter is a created by freezing the current model
weights, then training an additional set of weights to act upon the base
model that allows the LLM to have its weights altered only by the
adapter letting it be a plug and play solution, The matematical
representation of this is at a high level:

#figure(image("images/Base_model_fine_tuning.png", width: 60%), caption: [Fine tuning diagram])

$ \min L(D; W.x + Δ W.x) $ <fine_tune_explained>


-   L : This represents the loss function used to calculate the
    gradient that needs to be minimized for the LLM.

-   D : This represents the dataset the loss function is being
    trained on and what is being optimized for.

-   Θ : θ₀ represent the weights for base model and Δθ represent the
    weights from the fine tuning. This is how the adapter is swappable
    as the weights are not integrated into the base model.

#figure(
  image("images/2025-10-22-12:39:26.png", width:60%)
  , caption: [Adapter pattern]
)
= Fine Tuning approaches

== Full Parameter Fine tuning

Full parameter fine tuning approach was first proposed in 2018 @howard_universal_2018 called ULMFiT. This principle has been taken and applied in many forms to models such as DistilBERT@sanh_distilbert_2020 and BERT @devlin_bert_2019.
The base approach is is freezing the original weights then creating a blank matrix of the model weights, then training those to scale each weight individually to bias towards the new target. 
While efficent it is still a computationally heavy process as every single weight is modified but as a baseline it allows for the adapter architecture to be used.

The formula used to represent this would be

$ \min L(D; W.x + Δ W.x) $


The size of matrix θ₀ and Δθ are both m \* n where m and n represent the rows and columns in θ₀. The rest of the definitions are here @fine_tune_explained.
This method of training allows a small dataset to impact the results of a larger model removing the need to train the model on a huge corpus of data to get tangible results.

== Lora Fine Tuning


== Vera Fine Tuning
Vera @kopiczko_vera_2024 fine tuning is an innovation on LoRa fine tuning created to reduce the memory overhead in LoRa.
The method in which it works is based on Random Matrix Adaptation and LoRa. 
The process begins by generating two low rank matrixes of sizes m \* 
In mathematical terms 

$ W.x + Δ W.x = W.x + Λ_b B Λ_d A x $

- A and B:  Are randomly generated low rank matrixes of sizes m \* r and r \* n which multiply to create the W.x matrix.

- Λ_b and  Λ_d: Are diagonal matrixes which are used to scale the A and B matrixes. They are of sizes m \* m and r \* r.


The key innovation to note is the $ Λ_b B Λ_d A x$. These are diagonal scaling matrixes which scale the values of the randomly gernated A and B matrixes which emulate 


== QLora Fine Tuning

== Lora vs Full fine tuning

== Lora Vs Vera

#bibliography("references.bib")
