# Word-Based-Tic-tac-toe
This project uses classic Tic-Tac-Toe game by replacing traditional Xs and Os with synonyms sourced from WordNet, adding a layer of semantic depth to gameplay. Leveraging reinforcement learning (Q-learning), the AI adapts its strategy dynamically based on the meaning of the words placed on the board. Key technical aspects include:

- WordNet Integration: Words are selected and compared using the WordNet lexical database to ensure semantic relevance.
- Natural Language Processing Tools: NLTK is used for tokenization and pre-processing, while Gensim's Word2Vec model calculates word similarities.
- Reinforcement Learning: Q-learning trains the AI to optimize gameplay strategies, with a reward-based system guiding decision-making.
- Semantic Similarity Measures: Wu-Palmer similarity and Word2Vec embeddings are employed to assess word relationships.
- Dynamic Gameplay: The AI continuously refines its moves by analyzing the board state and word semantics to outmaneuver opponents.
This innovative approach bridges language understanding with strategic gaming, offering an engaging platform for exploring AI-driven gameplay.
