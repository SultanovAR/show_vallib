

class TableValidation(BaseValidation):
    def __init__(
        self,
        model,
        sampler,
        scorer,
        pipeline=pipeline_31,
        exclude_tests=[],
        custom_tests={},
        # **kwargs,
    ):
        super().__init__(
            model, sampler, scorer, pipeline, exclude_tests, custom_tests
        )
        
        if isinstance(pipeline, str):
            self.pipeline = self._parse_pipeline()
        else:
            self.pipeline = pipeline
            
        self.pipeline['tests'].update(custom_tests)
        self.pipeline['tests'] = {self.pipeline['tests'][key] for key in self.pipeline['tests'] if key not in exclude_tests}