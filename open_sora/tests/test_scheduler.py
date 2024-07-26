z = np.random.normal(size=[1, 4, 30, 90, 160])
masks = np.ones((1, 30))

scheduler.sample(
    model,
    text_encoder,
    z=z,
    prompts=batch_prompts_loop,
    additional_args=model_args,
    masks=masks,
)
