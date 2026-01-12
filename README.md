MULTI MODAL RAG
Multi Modal RAG (Retrieval-Augmented Generation) is an AI system that combines text, images,
and documents with retrieval and large language models to generate accurate, context-aware
responses.
Project Objective
• Combine multiple data modalities
• Retrieve relevant context
• Generate grounded AI responses

def search_images(query_text, top_k=5):
    # Process the text query
    inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity scores
    similarity = torch.mm(text_features, image_emb.T)
    
    # Get top-k matches
    values, indices = similarity[0].topk(min(top_k, len(data)))
    
    # Display results
    print(f"\nSearch query: {query_text}")
    print("\nTop matches:")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, top_k, figsize=(15, 3))
    
    for i, (idx, score) in enumerate(zip(indices, values)):
        # Print text and score
        print(f"{data['text'][idx]}: {score:.3f}")
        
        # Display image
        axes[i].imshow(data['image'][idx])
        axes[i].axis('off')
        axes[i].set_title(f"Score: {score:.3f}")
    
    plt.tight_layout()
    plt.show()


