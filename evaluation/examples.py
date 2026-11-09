"""Eval examples"""
EVAL_EXAMPLES = [
    {
        "question": "What is the main purpose of the system described in the document?",
        "answer": "The main purpose of the Internet is to act as a networking infrastructure that provides services to distributed applications (such as web, email, video streaming, and social networks) and allows end systems (hosts) to exchange data reliably and efficiently across the globe.",
    },
    {
        "question": "What are the key components mentioned in the document?",
        "answer": "The key components of the network include end systems (or hosts, like PCs, smartphones, and servers), communication links (such as coaxial cable, copper wire, optical fiber, and radio spectrum), packet switches (primarily routers and link-layer switches), and Internet Service Providers (ISPs) that connect them all together.",
    },
    {
        "question": "What protocols or standards does the document reference?",
        "answer": "The text references Internet standards developed by the IETF (Internet Engineering Task Force) documented as RFCs (Requests for Comments). Key protocols include HTTP, SMTP, and DNS at the application layer; TCP and UDP at the transport layer; IP, OSPF, and BGP at the network layer; and Ethernet and IEEE 802.11 (Wi-Fi) at the link layer.",
    },
]

DATASET_NAME = "RAG-eval"
