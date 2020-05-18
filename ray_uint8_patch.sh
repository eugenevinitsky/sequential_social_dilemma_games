# Find Python folder name so that this patch can run correctly on different versions of Python.
python_folder_name=$(ls venv/lib)

# Apply patches
sed -i '110s/tf.float32/tf.uint8/' venv/lib/"$python_folder_name"/site-packages/ray/rllib/policy/dynamic_tf_policy.py # Hardcoded observation space to uint8.
sed -i '76s/np.float32/np.uint8/' venv/lib/"$python_folder_name"/site-packages/ray/rllib/models/preprocessors.py # Same as above.
sed -i '231s/np.zeros(self.shape)/np.zeros(self.shape, dtype=self.observation_space.dtype)/' venv/lib/"$python_folder_name"/site-packages/ray/rllib/models/preprocessors.py # Change observation shape to what we actually provide
sed -i '215s/tf.int64/action_space.dtype/' venv/lib/"$python_folder_name"/site-packages/ray/rllib/models/catalog.py # Change action shape to what we actually provide
sed -i '55s/tf.math.argmax(self.inputs, axis=1)/tf.math.argmax(self.inputs, axis=1, output_type=tf.int32)/' venv/lib/"$python_folder_name"/site-packages/ray/rllib/models/tf/tf_action_dist.py # Actions should not sample at int64, int32 is the lowest that multinomial takes
sed -i '83s/tf.multinomial(self.inputs, 1)/tf.multinomial(self.inputs, 1, output_dtype=tf.int32)/' venv/lib/"$python_folder_name"/site-packages/ray/rllib/models/tf/tf_action_dist.py # Same as above
sed -i '644i        actions = np.array(actions, dtype=policy.action_space.dtype)' venv/lib/"$python_folder_name"/site-packages/ray/rllib/evaluation/sampler.py # Insert action to uint8 conversion to save even more memory
