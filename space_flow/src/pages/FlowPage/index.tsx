import { useCallback, useContext, useEffect, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  addEdge,
  useEdgesState,
  useNodesState,
} from "reactflow";
import TextUpdaterNode from "../../CustomNodes/inputTextFolder";
import PromptNode from "../../CustomNodes/PromptNode";
import ModelNode from "../../CustomNodes/ModelNode";
import { locationContext } from "../../contexts/locationContext";
import { ExtraSidebar } from "./components/extraSidebarComponent";

export default function FlowPage() {
  // outside component to avoid render trigger

  const reactFlowWrapper = useRef(null);
  const nodeTypes = {
    textUpdater: TextUpdaterNode,
    promptNode: PromptNode,
    modelNode: ModelNode,
  };

  const rfStyle = {
    backgroundCOlor: "#B8CEFF",
  };
  let id = 0;
  const getId = () => `dndnode_${id++}`;

  const { setExtraComponent } = useContext(locationContext);

  useEffect(() => {
    setExtraComponent(ExtraSidebar);
  }, []);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    []
  );
  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);
  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const reactflowBounds = reactFlowWrapper.current.getBoundingClientRect();
      const type = event.dataTransfer.getData("application/reactflow");
      let data = JSON.parse(event.dataTransfer.getData("json"));
      // check if the dropped element is valid
      if (typeof type === "undefined" || !type) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactflowBounds.left,
        y: event.clientY - reactflowBounds.top,
      });
      const newNode = {
        id: getId(),
        type,
        position,
        data: { ...data, delete: () => console.log("asdsdsadad") },
      };
      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance]
  );

  return (
    <div className="w-full h-full" ref={reactFlowWrapper}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={setReactFlowInstance}
        nodeTypes={nodeTypes}
        onDragOver={onDragOver}
        onDrop={onDrop}
        fitView
      >
        <Background />
        <Controls></Controls>
      </ReactFlow>
    </div>
  );
}