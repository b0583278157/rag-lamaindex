from llama_index.core.workflow import Workflow, step, Context, StartEvent, StopEvent


class RAGWorkflow(Workflow):

    def __init__(self, pipeline):
        super().__init__()
        self.retriever = pipeline["retriever"]
        self.llm = pipeline["llm"]

    # 1️⃣ Start
    @step
    def start(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        return QueryEvent(query=ev.query)

    # 2️⃣ Retrieve
    @step
    def retrieve(self, ctx: Context, ev: QueryEvent) -> RetrievedEvent:
        nodes = self.retriever.retrieve(ev.query)

        return RetrievedEvent(
            query=ev.query,
            nodes=nodes
        )

    # 3️⃣ Validate (קריטי למטלה)
    @step
    def validate(self, ctx: Context, ev: RetrievedEvent) -> ValidateEvent:

        nodes = ev.nodes

        # בדיקות תקינות בסיסיות
        if not nodes:
            return ValidateEvent(
                query=ev.query,
                nodes=[],
                is_valid=False,
                reason="No retrieval results"
            )

        # בדיקת איכות בסיסית (פשוטה אך מספיקה למטלה)
        if len(nodes) < 2:
            return ValidateEvent(
                query=ev.query,
                nodes=nodes,
                is_valid=False,
                reason="Too few results"
            )

        return ValidateEvent(
            query=ev.query,
            nodes=nodes,
            is_valid=True
        )

    # 4️⃣ Routing step (Event-driven decision)
    @step
    def route(self, ctx: Context, ev: ValidateEvent):

        # ❌ אם לא תקין → חיפוש מחדש (אותו flow, לא משנה פונקציונליות)
        if not ev.is_valid:
            nodes = self.retriever.retrieve(ev.query + " ")  # retry קטן

            return RetrievedEvent(
                query=ev.query,
                nodes=nodes
            )

        # ✅ אם תקין → המשך יצירה
        return GenerateEvent(
            query=ev.query,
            nodes=ev.nodes
        )

    # 5️⃣ Generate (RAG אמיתי)
    @step
    def generate(self, ctx: Context, ev: GenerateEvent) -> StopEvent:

        context_text = "\n\n".join(
            [n.get_content() for n in ev.nodes]
        )

        prompt = f"""
        Answer the question using the context below:

        Context:
        {context_text}

        Question:
        {ev.query}
        """

        response = self.llm.complete(prompt)

        return StopEvent(result=str(response))