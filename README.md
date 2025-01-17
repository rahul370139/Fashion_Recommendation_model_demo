**Title: Multimodal Recommendation System Using Image and Text Analysis for E-Commerce Applications**

Author: Rahul Sharma
Research question:
The goal of this project is to develop a multimodal recommendation system for the fashion industry. Users on e-commerce platforms often face challenges in finding products that match their exact preferences, such as liking a garment’s style but not its color or fabric. This project explores whether leveraging both image and text inputs can enhance recommendations to closely align with users’ desired attributes.
Importance:
Fashion e-commerce platforms host thousands of items, making product discovery overwhelming for users. An intuitive multimodal search tool—considering both image and textual descriptions—can significantly improve user experience and streamline product discovery. This innovation could lead to better customer satisfaction and higher conversion rates.
Problem Formulation:
The problem is framed as a similarity-based retrieval task. The goal is to retrieve visually and semantically similar items from a database based on user-provided inputs. This approach does not involve predicting labels but focuses on finding similar items within a shared embedding space created using a multimodal model.
Which ML task did you use?
The task is unsupervised similarity-based retrieval, combining image and text embeddings for similarity computation. This approach leverages both modalities to generate more relevant recommendations.

Dataset
Describe the dataset
The dataset used is DeepFashion1, which contains approximately 44,000 high-quality fashion images in PNG and JPG format, each with a resolution of 512×1024 and 750×1101 respectively. These images cover a wide variety of clothing items, fabrics, and textures. While labels such as fabric type and texture are provided, they are not fully aligned with the goals of capturing users’ preferences for color or pattern. Thus, the focus is primarily on the images themselves.
What was the modality, size, and features?
Modality: Images
Size: Approximately 44,000 samples
Features: Visual features of clothing items, including style, texture, and color
Labels: Fabric type, texture (not used for this project)
How was the data collected?
The dataset was sourced from a public GitHub repository associated with the DeepFashion dataset.
Importance of the dataset
DeepFashion is widely recognized for its high-quality, diverse images, making it suitable for tasks involving fashion-related research. Its variety allows models like CLIP to extract meaningful embeddings that effectively capture attributes such as style, pattern, and color.

ML Methodology
Methods Used
The project employs the CLIP model, a pre-trained multimodal model designed to align images and text in a shared embedding space. This alignment enables similarity computation across modalities, facilitating the retrieval of items matching both visual and textual inputs.
How does CLIP work?
CLIP (Contrastive Language-Image Pretraining) is a model trained on large-scale image-text pairs. It consists of:
Vision Transformer (ViT): Processes images by encoding them into embeddings through a multi-layered transformer architecture. The process involves:
A convolutional layer (Conv2D) to extract features.
Residual attention blocks that combine multi-head attention and feed-forward layers for hierarchical feature learning.
Layer normalization to stabilize learning.
Text Transformer: Encodes text inputs into embeddings using token embeddings, positional encodings, and transformer layers. It uses layer normalization and a final embedding space projection.
Shared Embedding Space: Both visual and textual embeddings are projected into the same space, enabling similarity computation between modalities. The contrastive objective during training ensures that image-text pairs with similar meanings are close in the shared space.
Preprocessing Steps
For each image in the dataset:
Loading the Image: Convert images to RGB format for standardization.
Image Segmentation and Masking: Apply segmentation techniques to isolate clothing items from backgrounds. Masked images focus the embeddings on the garments, reducing the influence of irrelevant features like accessories or backgrounds.
CLIP Preprocessing: Apply transformations such as resizing, center-cropping, and normalization to create 224×224 tensors.
Encoding the Image: Use the CLIP image encoder to generate normalized embeddings representing the semantic content.
How was the dataset used?
The dataset images were pre-encoded into embeddings using CLIP’s image encoder and stored for efficient similarity computations at inference time.
Data Splits
No explicit training/testing split was required, as this is a retrieval task. The entire dataset serves as a searchable corpus, while user inputs (images and/or text) act as queries during inference.
Missing Data and Cleaning
There were no significant missing data issues. All images were usable for the task.
Frameworks and Libraries
PyTorch: For implementing and running CLIP
CLIP (OpenAI): For embedding generation
scikit-learn: For cosine similarity computation
Environment: Personal computer (Mac, Apple MPS backend), Python-based environment

Results
Procedure
Embedding the Dataset:
Precomputed embeddings for all ∼44k images using the CLIP image encoder and stored them for fast retrieval.
User Query:
Only an image query: The user-provided image is encoded, and its embedding is compared to the precomputed dataset embeddings to find nearest neighbors.
Image + Text query: Both inputs are encoded separately using CLIP’s image and text encoders, and their embeddings are combined. Initial weights for the combination were equal, but further refinements prioritized textual embeddings.
Combining Embeddings:
CLIP aligns image and text embeddings into the same space, making it feasible to combine them. Initially, equal weights were assigned to image and text embeddings, but results favored images over text.
Reranking Approach:
To address this imbalance, a two-stage retrieval was implemented:
First Pass: Equal weights were used to retrieve top-k results.
Second Pass: The initial results were reranked by assigning a 90% weight to text embeddings and 10% to image embeddings, ensuring textual descriptions had greater influence.
Modeling Attempts:
Alternative models like BLIP were tested but underperformed compared to CLIP in capturing fine-grained fashion attributes. Thus, CLIP was retained as the primary model.


Observations
Equal Weightage Results: Visually similar items were retrieved but lacked alignment with textual nuances.
Text-Weighted Reranking: Improved alignment with user queries, as indicated by qualitative feedback.
Manual Evaluation: Without ground-truth relevance judgments, we evaluated performance by manual inspection. A small set of queries (images + optional text) was prepared and shown to a group of test users. We created a Google Form to let them rate how closely the recommended images matched their intended criteria.
Questions included: 
“Does the returned set of images match the style/pattern you had in mind?” 
“Rate on a scale of 1-5 how satisfied you are with the recommendations.”

Lessons Learned
What did you learn?
Simplicity in Approaches: Straightforward solutions, like reweighting embeddings, can significantly improve performance without adding complexity.
Balancing Modalities: While CLIP’s joint embedding space is powerful, careful balancing of image and text contributions is crucial for optimal results.
Importance of Evaluation: User feedback highlighted the effectiveness of the reranking approach, compensating for the absence of ground-truth relevance labels.
Challenges
Modality Balance: Achieving the right weight distribution between image and text embeddings required iterative experimentation.
Subjective Evaluation: The lack of annotated ground truth made it necessary to rely on user surveys, which are subjective and time-consuming.
Key Takeaways
Readers should understand that combining embeddings in a shared space, supported by techniques like reranking, can greatly enhance multimodal retrieval tasks. Balancing simplicity and performance is often the key to effective solutions.

Conclusion
Final Remarks
This project developed a multimodal recommendation system leveraging the CLIP model to align image and text queries within the fashion domain. By embedding these modalities into a shared space, we enabled effective similarity-based retrieval. Introducing segmentation and reranking further refined the results, enhancing alignment with user preferences.
Future Work
Fine-tune CLIP with domain-specific image-text pairs for improved alignment.
Automate evaluation by collecting annotated relevance labels.
Explore learnable combination layers for embeddings to optimize performance.
Try various other methods and find recommendation-generating captions as well.

References
Radford, A. et al. "Learning Transferable Visual Models From Natural Language Supervision (CLIP)."
DeepFashion Dataset: Liu, Z. et al. "DeepFashion: Powering robust clothes recognition and retrieval with rich annotations." CVPR 2016.

